import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app import database
from app.database import Base
from app.models import (
    EvaluationEpisode,
    Experiment,
    PolicyEvaluation,
    TrainingMetric,
    TrainingRun,
)
from app.routers import comparison, experiments
from app.schemas.comparison import ExperimentComparisonRequest
from app.schemas.experiment import ExperimentCreate, ExperimentResponse
from app.schemas.runs import EvaluationRequest, PolicyEvaluationResponse
from app.services import evaluation_service, training_service
from app.services import training_metrics


class PhaseSixTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            f"sqlite:///{(Path(self.temp_dir.name) / 'phase6.db').as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )
        with self.session_factory() as db:
            self.experiment = Experiment(
                name="Seeded experiment",
                environment="InvertedPendulum-v5",
                algorithm="PPO",
                total_timesteps=32,
                status="created",
                base_seed=42,
                run_count=2,
                learning_rate=3e-4,
                n_steps=8,
                batch_size=4,
                gamma=0.99,
                n_epochs=2,
            )
            self.other_experiment = Experiment(
                name="No metrics",
                environment="InvertedPendulum-v5",
                algorithm="PPO",
                total_timesteps=32,
                status="created",
                base_seed=50,
                run_count=1,
                learning_rate=3e-4,
                n_steps=8,
                batch_size=4,
                gamma=0.99,
                n_epochs=2,
            )
            db.add_all([self.experiment, self.other_experiment])
            db.commit()
            self.experiment_id = self.experiment.id
            self.other_experiment_id = self.other_experiment.id

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()
        for experiment_id in range(1, 20):
            training_service.release_job(experiment_id)

    def test_new_schema_contains_run_and_evaluation_tables(self):
        inspector = inspect(self.engine)
        self.assertIn("training_runs", inspector.get_table_names())
        self.assertIn("policy_evaluations", inspector.get_table_names())
        self.assertIn("evaluation_episodes", inspector.get_table_names())
        self.assertIn("run_id", {item["name"] for item in inspector.get_columns("training_metrics")})

    def test_seed_configuration_validation_and_repeat_run_seed_sequence(self):
        with self.assertRaises(ValidationError):
            ExperimentCreate.model_validate(
                {"name": "Invalid", "base_seed": -1, "run_count": 2}
            )
        with self.assertRaises(ValidationError):
            ExperimentCreate.model_validate(
                {"name": "Invalid", "base_seed": 1, "run_count": 21}
            )

        with (
            self.session_factory() as db,
            patch.object(experiments, "launch_job") as launch,
        ):
            experiment = db.get(Experiment, self.experiment_id)
            experiment.base_seed = 4_294_967_295
            response = experiments.start_experiment(self.experiment_id, db)
            runs = (
                db.query(TrainingRun)
                .filter_by(experiment_id=self.experiment_id)
                .order_by(TrainingRun.run_index)
                .all()
            )

        self.assertEqual(response.status, "pending")
        self.assertEqual([run.run_index for run in runs], [0, 1])
        self.assertEqual([run.effective_seed for run in runs], [4_294_967_295, 0])
        self.assertEqual([run.status for run in runs], ["pending", "pending"])
        launch.assert_called_once()

    def test_run_metrics_are_scoped_and_ordered_by_timestep(self):
        with self.session_factory() as db:
            first_run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=0,
                effective_seed=42,
                status="completed",
                requested_timesteps=32,
            )
            second_run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=1,
                effective_seed=43,
                status="completed",
                requested_timesteps=32,
            )
            db.add_all([first_run, second_run])
            db.flush()
            db.add_all(
                [
                    TrainingMetric(
                        experiment_id=self.experiment_id,
                        run_id=first_run.id,
                        global_timestep=20,
                        completed_episode_count=1,
                        episode_reward=5,
                        episode_length=2,
                        recent_mean_episode_reward=5,
                        elapsed_seconds=2,
                        timesteps_per_second=10,
                        training_status="completed",
                    ),
                    TrainingMetric(
                        experiment_id=self.experiment_id,
                        run_id=first_run.id,
                        global_timestep=10,
                        completed_episode_count=1,
                        episode_reward=3,
                        episode_length=1,
                        recent_mean_episode_reward=3,
                        elapsed_seconds=1,
                        timesteps_per_second=10,
                        training_status="running",
                    ),
                    TrainingMetric(
                        experiment_id=self.experiment_id,
                        run_id=second_run.id,
                        global_timestep=99,
                        completed_episode_count=1,
                        episode_reward=999,
                        episode_length=99,
                        recent_mean_episode_reward=999,
                        elapsed_seconds=1,
                        timesteps_per_second=99,
                        training_status="completed",
                    ),
                ]
            )
            db.commit()
            first_run_id = first_run.id

            records = experiments.get_run_metrics(
                self.experiment_id,
                first_run_id,
                10,
                db,
            )

        self.assertEqual([record.global_timestep for record in records], [10, 20])
        self.assertTrue(all(record.run_id == first_run_id for record in records))

    def test_evaluation_uses_reproducible_seed_sequence_and_persists_actual_episodes(self):
        with self.session_factory() as db:
            run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=0,
                effective_seed=42,
                status="completed",
                requested_timesteps=32,
                actual_timesteps=32,
            )
            db.add(run)
            db.flush()
            evaluation = PolicyEvaluation(
                run_id=run.id,
                status="pending",
                episode_count=2,
                base_seed=100_000,
            )
            db.add(evaluation)
            db.commit()
            run_id = run.id
            evaluation_id = evaluation.id

        class FakeEnvironment:
            def __init__(self):
                self.seeds = []

            def reset(self, *, seed):
                self.seeds.append(seed)
                return 0, {}

            def step(self, _action):
                return 0, 1.5, True, False, {}

            def close(self):
                return None

        class FakeModel:
            def predict(self, observation, deterministic):
                self.assert_deterministic = deterministic
                return 0, None

        fake_environment = FakeEnvironment()
        artifact_dir = Path(self.temp_dir.name) / "models"
        artifact_dir.mkdir()
        (artifact_dir / f"run_{run_id}.zip").touch()
        with (
            patch.object(evaluation_service, "SessionLocal", self.session_factory),
            patch.object(evaluation_service.gym, "make", return_value=fake_environment),
            patch.object(evaluation_service.PPO, "load", return_value=FakeModel()),
            patch.object(evaluation_service, "MODEL_DIR", artifact_dir),
        ):
            evaluation_service.evaluate_policy(evaluation_id)

        with self.session_factory() as db:
            result = db.get(PolicyEvaluation, evaluation_id)
            episodes = (
                db.query(EvaluationEpisode)
                .filter_by(evaluation_id=evaluation_id)
                .order_by(EvaluationEpisode.episode_index)
                .all()
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.mean_episode_reward, 1.5)
        self.assertEqual(result.mean_episode_length, 1)
        self.assertEqual([episode.seed for episode in episodes], [100_000, 100_001])
        self.assertEqual([episode.reward for episode in episodes], [1.5, 1.5])
        self.assertEqual([episode.length for episode in episodes], [1, 1])

    def test_evaluation_failure_is_persisted_without_fabricated_results(self):
        with self.session_factory() as db:
            run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=0,
                effective_seed=42,
                status="completed",
                requested_timesteps=32,
            )
            db.add(run)
            db.flush()
            evaluation = PolicyEvaluation(
                run_id=run.id,
                status="pending",
                episode_count=2,
                base_seed=100_000,
            )
            db.add(evaluation)
            db.commit()
            evaluation_id = evaluation.id

        with (
            patch.object(evaluation_service, "SessionLocal", self.session_factory),
            patch.object(
                evaluation_service,
                "MODEL_DIR",
                Path(self.temp_dir.name) / "missing-models",
            ),
        ):
            evaluation_service.evaluate_policy(evaluation_id)

        with self.session_factory() as db:
            result = db.get(PolicyEvaluation, evaluation_id)
            self.assertEqual(result.status, "failed")
            self.assertIsNone(result.mean_episode_reward)
            self.assertEqual(result.episodes, [])

    def test_evaluation_endpoint_validates_run_and_artifact_and_queues_job(self):
        with self.session_factory() as db:
            with self.assertRaises(HTTPException) as missing:
                experiments.start_run_evaluation(
                    999,
                    EvaluationRequest(),
                    db,
                )
            self.assertEqual(missing.exception.status_code, 404)

            run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=0,
                effective_seed=42,
                status="running",
                requested_timesteps=32,
            )
            db.add(run)
            db.commit()
            run_id = run.id
            with self.assertRaises(HTTPException) as active:
                experiments.start_run_evaluation(
                    run_id,
                    EvaluationRequest(),
                    db,
                )
            self.assertEqual(active.exception.status_code, 409)

            run.status = "completed"
            db.commit()
            with patch.object(
                experiments,
                "MODEL_DIR",
                Path(self.temp_dir.name) / "missing-models",
            ):
                with self.assertRaises(HTTPException) as no_artifact:
                    experiments.start_run_evaluation(
                        run_id,
                        EvaluationRequest(),
                        db,
                    )
            self.assertEqual(no_artifact.exception.status_code, 409)

            model_dir = Path(self.temp_dir.name) / "models"
            model_dir.mkdir()
            (model_dir / f"run_{run_id}.zip").touch()
            with (
                patch.object(experiments, "MODEL_DIR", model_dir),
                patch.object(experiments, "launch_evaluation") as launch,
            ):
                evaluation = experiments.start_run_evaluation(
                    run_id,
                    EvaluationRequest(episode_count=2, base_seed=123),
                    db,
                )
                self.assertEqual(evaluation.status, "pending")
                launch.assert_called_once()
                with self.assertRaises(HTTPException) as duplicate:
                    experiments.start_run_evaluation(
                        run_id,
                        EvaluationRequest(),
                        db,
                    )
            self.assertEqual(duplicate.exception.status_code, 409)

    def test_comparison_reports_sample_stddev_across_evaluated_run_means(self):
        with self.session_factory() as db:
            first_run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=0,
                effective_seed=42,
                status="completed",
                requested_timesteps=32,
            )
            second_run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=1,
                effective_seed=43,
                status="completed",
                requested_timesteps=32,
            )
            db.add_all([first_run, second_run])
            db.flush()
            evaluations = [
                PolicyEvaluation(
                    run_id=first_run.id,
                    status="completed",
                    episode_count=2,
                    base_seed=100_000,
                    mean_episode_reward=100,
                    mean_episode_length=25,
                ),
                PolicyEvaluation(
                    run_id=second_run.id,
                    status="completed",
                    episode_count=1,
                    base_seed=100_000,
                    mean_episode_reward=80,
                    mean_episode_length=10,
                ),
            ]
            db.add_all(evaluations)
            db.flush()
            db.add_all(
                [
                    EvaluationEpisode(
                        evaluation_id=evaluations[0].id,
                        episode_index=0,
                        seed=100_000,
                        reward=90,
                        length=20,
                    ),
                    EvaluationEpisode(
                        evaluation_id=evaluations[0].id,
                        episode_index=1,
                        seed=100_001,
                        reward=110,
                        length=30,
                    ),
                    EvaluationEpisode(
                        evaluation_id=evaluations[1].id,
                        episode_index=0,
                        seed=100_000,
                        reward=80,
                        length=10,
                    ),
                ]
            )
            db.commit()
            request = ExperimentComparisonRequest(
                experiment_ids=[self.experiment_id, self.other_experiment_id]
            )
            result = comparison.compare_experiments(request, db)

        item, empty_item = result.experiments
        self.assertEqual(item.training_run_count, 2)
        self.assertEqual(item.completed_training_run_count, 2)
        self.assertEqual(item.evaluated_run_count, 2)
        self.assertEqual(item.evaluation_reward_mean, 90)
        self.assertAlmostEqual(item.evaluation_reward_sample_stddev, 14.1421356237)
        self.assertEqual(item.evaluation_episode_count, 3)
        self.assertEqual(item.mean_evaluation_episode_length, 20)
        self.assertEqual(empty_item.evaluated_run_count, 0)
        self.assertIsNone(empty_item.evaluation_reward_mean)
        self.assertIsNone(empty_item.evaluation_reward_sample_stddev)

    def test_single_evaluated_run_has_no_between_run_standard_deviation(self):
        with self.session_factory() as db:
            run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=0,
                effective_seed=42,
                status="completed",
                requested_timesteps=32,
            )
            db.add(run)
            db.flush()
            db.add(
                PolicyEvaluation(
                    run_id=run.id,
                    status="completed",
                    episode_count=1,
                    base_seed=100_000,
                    mean_episode_reward=12.0,
                    mean_episode_length=20.0,
                )
            )
            db.commit()
            result = comparison.compare_experiments(
                ExperimentComparisonRequest(
                    experiment_ids=[
                        self.experiment_id,
                        self.other_experiment_id,
                    ]
                ),
                db,
            )

        item = result.experiments[0]
        self.assertEqual(item.evaluated_run_count, 1)
        self.assertEqual(item.evaluation_reward_mean, 12.0)
        self.assertIsNone(item.evaluation_reward_sample_stddev)

    def test_run_response_includes_saved_seed_and_evaluation_details(self):
        with self.session_factory() as db:
            run = TrainingRun(
                experiment_id=self.experiment_id,
                run_index=0,
                effective_seed=44,
                status="completed",
                requested_timesteps=32,
            )
            db.add(run)
            db.flush()
            run_id = run.id
            response = ExperimentResponse.model_validate(
                db.get(Experiment, self.experiment_id)
            )

        self.assertEqual(response.base_seed, 42)
        self.assertEqual(response.run_count, 2)
        self.assertEqual(response.runs[0].id, run_id)
        self.assertEqual(response.runs[0].effective_seed, 44)

    def test_worker_runs_seeded_jobs_sequentially_and_tags_metrics(self):
        with self.session_factory() as db:
            experiment = db.get(Experiment, self.experiment_id)
            experiment.status = "pending"
            db.commit()
            for run_index in range(2):
                db.add(
                    TrainingRun(
                        experiment_id=self.experiment_id,
                        run_index=run_index,
                        effective_seed=42 + run_index,
                        status="pending",
                        requested_timesteps=32,
                    )
                )
            db.commit()

        captured_seeds = []
        active_models = 0
        maximum_active_models = 0

        class FakeEnvironment:
            def close(self):
                return None

        class FakeModel:
            def __init__(self, *_args, seed, **_kwargs):
                nonlocal active_models, maximum_active_models
                captured_seeds.append(seed)
                active_models += 1
                maximum_active_models = max(maximum_active_models, active_models)

            def learn(self, *, total_timesteps, callback):
                callback.model = object()
                callback.num_timesteps = total_timesteps
                callback._on_training_start()
                callback.locals = {"infos": []}
                callback._on_step()
                self.num_timesteps = total_timesteps

            def save(self, _path):
                nonlocal active_models
                active_models -= 1

        with (
            patch.object(training_service, "SessionLocal", self.session_factory),
            patch.object(training_metrics, "SessionLocal", self.session_factory),
            patch.object(training_service.gym, "make", return_value=FakeEnvironment()),
            patch.object(training_service, "PPO", FakeModel),
            patch.object(training_service, "MODEL_DIR", Path(self.temp_dir.name)),
        ):
            training_service.run_experiment(self.experiment_id, threading.Event())

        with self.session_factory() as db:
            runs = (
                db.query(TrainingRun)
                .filter_by(experiment_id=self.experiment_id)
                .order_by(TrainingRun.run_index)
                .all()
            )
            metrics = (
                db.query(TrainingMetric)
                .filter(TrainingMetric.experiment_id == self.experiment_id)
                .order_by(TrainingMetric.id)
                .all()
            )
            experiment = db.get(Experiment, self.experiment_id)
        self.assertEqual(captured_seeds, [42, 43])
        self.assertEqual(maximum_active_models, 1)
        self.assertEqual([run.status for run in runs], ["completed", "completed"])
        self.assertEqual(experiment.status, "completed")
        self.assertEqual({metric.run_id for metric in metrics}, {run.id for run in runs})

    def test_failed_run_does_not_prevent_later_runs_and_experiment_is_failed(self):
        with self.session_factory() as db:
            experiment = db.get(Experiment, self.experiment_id)
            experiment.status = "pending"
            db.commit()
            for run_index in range(2):
                db.add(
                    TrainingRun(
                        experiment_id=self.experiment_id,
                        run_index=run_index,
                        effective_seed=70 + run_index,
                        status="pending",
                        requested_timesteps=32,
                    )
                )
            db.commit()

        attempted_seeds = []

        class FakeEnvironment:
            def close(self):
                return None

        class FakeModel:
            def __init__(self, *_args, seed, **_kwargs):
                self.seed = seed
                attempted_seeds.append(seed)

            def learn(self, *, total_timesteps, callback):
                if self.seed == 70:
                    raise RuntimeError("private model details")
                callback.model = object()
                callback.num_timesteps = total_timesteps
                callback._on_training_start()
                callback.locals = {"infos": []}
                callback._on_step()
                self.num_timesteps = total_timesteps

            def save(self, _path):
                return None

        with (
            patch.object(training_service, "SessionLocal", self.session_factory),
            patch.object(training_metrics, "SessionLocal", self.session_factory),
            patch.object(training_service.gym, "make", return_value=FakeEnvironment()),
            patch.object(training_service, "PPO", FakeModel),
            patch.object(training_service, "MODEL_DIR", Path(self.temp_dir.name)),
        ):
            training_service.run_experiment(self.experiment_id, threading.Event())

        with self.session_factory() as db:
            runs = (
                db.query(TrainingRun)
                .filter_by(experiment_id=self.experiment_id)
                .order_by(TrainingRun.run_index)
                .all()
            )
            experiment = db.get(Experiment, self.experiment_id)
        self.assertEqual(attempted_seeds, [70, 71])
        self.assertEqual([run.status for run in runs], ["failed", "completed"])
        self.assertIn("RuntimeError", runs[0].failure_message)
        self.assertNotIn("private model details", runs[0].failure_message)
        self.assertEqual(experiment.status, "failed")

    def test_legacy_database_migration_preserves_experiment_and_metrics(self):
        legacy_path = Path(self.temp_dir.name) / "legacy-phase6.db"
        legacy_engine = create_engine(f"sqlite:///{legacy_path.as_posix()}")
        with legacy_engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE experiments ("
                "id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, "
                "environment VARCHAR NOT NULL, algorithm VARCHAR NOT NULL, "
                "status VARCHAR NOT NULL, failure_message TEXT, "
                "total_timesteps INTEGER NOT NULL, learning_rate FLOAT, "
                "n_steps INTEGER, batch_size INTEGER, gamma FLOAT, n_epochs INTEGER, "
                "created_at DATETIME NOT NULL, completed_at DATETIME)"
            )
            connection.exec_driver_sql(
                "INSERT INTO experiments VALUES "
                "(5,'Legacy','InvertedPendulum-v5','PPO','completed',NULL,32,"
                "0.0003,8,4,0.99,2,'2026-01-01',NULL)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE training_metrics ("
                "id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL, "
                "global_timestep INTEGER NOT NULL, completed_episode_count INTEGER NOT NULL, "
                "episode_reward FLOAT, episode_length INTEGER, "
                "recent_mean_episode_reward FLOAT, elapsed_seconds FLOAT NOT NULL, "
                "timesteps_per_second FLOAT, training_status VARCHAR NOT NULL, "
                "timestamp DATETIME NOT NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO training_metrics VALUES "
                "(7,5,20,1,12.5,20,12.5,1.0,20.0,'completed','2026-01-01')"
            )

        with patch.object(database, "engine", legacy_engine):
            database.initialize_database()

        with legacy_engine.connect() as connection:
            experiment_row = connection.exec_driver_sql(
                "SELECT id,name,base_seed,run_count FROM experiments WHERE id=5"
            ).one()
            metric_row = connection.exec_driver_sql(
                "SELECT id,experiment_id,global_timestep,episode_reward,run_id "
                "FROM training_metrics WHERE id=7"
            ).one()
            tables = set(inspect(legacy_engine).get_table_names())
        self.assertEqual(experiment_row, (5, "Legacy", None, None))
        self.assertEqual(metric_row, (7, 5, 20, 12.5, None))
        self.assertTrue(
            {"training_runs", "policy_evaluations", "evaluation_episodes"}.issubset(tables)
        )
        legacy_engine.dispose()

    def test_evaluation_request_is_bounded(self):
        for payload in ({"episode_count": 0}, {"episode_count": 101}, {"base_seed": -1}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                EvaluationRequest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
