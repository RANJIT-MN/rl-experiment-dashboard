import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    EvaluationEpisode,
    Experiment,
    PolicyEvaluation,
    TrainingMetric,
    TrainingRun,
)
from app.routers.experiments import stop_experiment
from app.services import evaluation_service, training_service, training_metrics
from app.services.compute_slot import compute_slot


class PhaseSevenReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            f"sqlite:///{(Path(self.temp_dir.name) / 'phase7.db').as_posix()}",
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
                name="Reliability test",
                environment="InvertedPendulum-v5",
                algorithm="PPO",
                total_timesteps=32,
                status="created",
                base_seed=7,
                run_count=1,
                learning_rate=3e-4,
                n_steps=8,
                batch_size=4,
                gamma=0.99,
                n_epochs=2,
            )
            db.add(self.experiment)
            db.commit()
            self.experiment_id = self.experiment.id

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()
        for experiment_id in range(1, 20):
            training_service.release_job(experiment_id)

    def _add_run(self, *, experiment_id=None, status="pending", index=0):
        with self.session_factory() as db:
            run = TrainingRun(
                experiment_id=experiment_id or self.experiment_id,
                run_index=index,
                effective_seed=7 + index,
                status=status,
                requested_timesteps=32,
            )
            db.add(run)
            db.commit()
            return run.id

    def test_restart_reconciliation_fails_only_active_work_and_preserves_history(self):
        active_id = self._add_run(status="running")
        completed_id = self._add_run(status="completed", index=1)
        with self.session_factory() as db:
            active_experiment = db.get(Experiment, self.experiment_id)
            active_experiment.status = "running"
            done_experiment = Experiment(
                name="Previously completed",
                environment="InvertedPendulum-v5",
                algorithm="PPO",
                total_timesteps=16,
                status="completed",
                base_seed=20,
                run_count=1,
            )
            db.add(done_experiment)
            db.flush()
            historical = TrainingRun(
                experiment_id=done_experiment.id,
                run_index=0,
                effective_seed=20,
                status="completed",
                requested_timesteps=16,
                actual_timesteps=16,
            )
            db.add(historical)
            db.flush()
            evaluation = PolicyEvaluation(
                run_id=historical.id,
                status="running",
                episode_count=2,
                base_seed=1000,
            )
            db.add(evaluation)
            db.flush()
            db.add(
                EvaluationEpisode(
                    evaluation_id=evaluation.id,
                    episode_index=0,
                    seed=1000,
                    reward=10,
                    length=5,
                )
            )
            db.add(
                TrainingMetric(
                    experiment_id=self.experiment_id,
                    run_id=active_id,
                    global_timestep=12,
                    completed_episode_count=1,
                    episode_reward=4,
                    episode_length=4,
                    recent_mean_episode_reward=4,
                    elapsed_seconds=1,
                    timesteps_per_second=12,
                    training_status="running",
                )
            )
            inconsistent_experiment = Experiment(
                name="Completed parent with active run",
                environment="InvertedPendulum-v5",
                algorithm="PPO",
                total_timesteps=16,
                status="completed",
                base_seed=30,
                run_count=1,
            )
            db.add(inconsistent_experiment)
            db.flush()
            inconsistent_run = TrainingRun(
                experiment_id=inconsistent_experiment.id,
                run_index=0,
                effective_seed=30,
                status="running",
                requested_timesteps=16,
            )
            db.add(inconsistent_run)
            db.commit()
            inconsistent_experiment_id = inconsistent_experiment.id
            inconsistent_run_id = inconsistent_run.id
            db.commit()
            done_experiment_id = done_experiment.id
            historical_run_id = historical.id
            evaluation_id = evaluation.id

        with patch.object(training_service, "SessionLocal", self.session_factory):
            training_service.reconcile_interrupted_work()

        with self.session_factory() as db:
            interrupted_run = db.get(TrainingRun, active_id)
            completed_run = db.get(TrainingRun, completed_id)
            active_experiment = db.get(Experiment, self.experiment_id)
            done_experiment = db.get(Experiment, done_experiment_id)
            historical_run = db.get(TrainingRun, historical_run_id)
            evaluation = db.get(PolicyEvaluation, evaluation_id)
            inconsistent_experiment = db.get(Experiment, inconsistent_experiment_id)
            inconsistent_run = db.get(TrainingRun, inconsistent_run_id)
            metrics = db.query(TrainingMetric).filter_by(run_id=active_id).all()
            episode_rows = db.query(EvaluationEpisode).filter_by(
                evaluation_id=evaluation_id
            ).all()

        self.assertEqual(interrupted_run.status, "failed")
        self.assertIn("outcome is unknown", interrupted_run.failure_message)
        self.assertEqual(active_experiment.status, "failed")
        self.assertEqual(completed_run.status, "completed")
        self.assertEqual(done_experiment.status, "completed")
        self.assertEqual(historical_run.status, "completed")
        self.assertEqual(evaluation.status, "failed")
        self.assertEqual(inconsistent_experiment.status, "failed")
        self.assertEqual(inconsistent_run.status, "failed")
        self.assertEqual(len(episode_rows), 1)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].episode_reward, 4)

    def test_restart_reconciliation_does_not_change_completed_work(self):
        run_id = self._add_run(status="completed")
        with self.session_factory() as db:
            experiment = db.get(Experiment, self.experiment_id)
            experiment.status = "completed"
            run = db.get(TrainingRun, run_id)
            run.actual_timesteps = 32
            db.commit()

        with patch.object(training_service, "SessionLocal", self.session_factory):
            training_service.reconcile_interrupted_work()

        with self.session_factory() as db:
            experiment = db.get(Experiment, self.experiment_id)
            run = db.get(TrainingRun, run_id)
        self.assertEqual(experiment.status, "completed")
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.actual_timesteps, 32)

    def test_repeated_stop_is_rejected_after_first_cooperative_request(self):
        self._add_run(status="pending")
        with self.session_factory() as db:
            db.query(Experiment).filter_by(id=self.experiment_id).update(
                {Experiment.status: "running"}
            )
            db.commit()
        event = training_service.reserve_job(self.experiment_id)
        with self.session_factory() as db:
            result = stop_experiment(self.experiment_id, db)
        self.assertTrue(event.is_set())
        self.assertEqual(result.status, "stopping")

        with self.session_factory() as db:
            with self.assertRaises(HTTPException) as repeated:
                stop_experiment(self.experiment_id, db)
        self.assertEqual(repeated.exception.status_code, 409)
        training_service.release_job(self.experiment_id)

    def test_queue_is_bounded_and_compute_slot_is_shared(self):
        with patch.object(training_service, "MAX_QUEUED_EXPERIMENTS", 1):
            training_service.reserve_job(self.experiment_id)
            with self.assertRaises(RuntimeError):
                training_service.reserve_job(self.experiment_id + 1)
        training_service.release_job(self.experiment_id)

        self.assertIs(training_service.compute_slot, evaluation_service.compute_slot)
        acquired = compute_slot.acquire(blocking=False)
        self.assertTrue(acquired)
        run_id = self._add_run(status="completed")
        with self.session_factory() as db:
            evaluation = PolicyEvaluation(
                run_id=run_id,
                status="pending",
                episode_count=1,
                base_seed=100,
            )
            db.add(evaluation)
            db.commit()
            evaluation_id = evaluation.id

        entered = threading.Event()
        with patch.object(
            evaluation_service,
            "_evaluate_policy_with_slot",
            side_effect=lambda *_args: entered.set(),
        ):
            worker = threading.Thread(
                target=evaluation_service.evaluate_policy,
                args=(evaluation_id,),
            )
            worker.start()
            time.sleep(0.15)
            self.assertFalse(entered.is_set())
            compute_slot.release()
            worker.join(2)
        self.assertTrue(entered.is_set())
        self.assertFalse(worker.is_alive())

    def test_evaluation_shutdown_cancels_and_joins_active_evaluation(self):
        run_id = self._add_run(status="completed")
        with self.session_factory() as db:
            evaluation = PolicyEvaluation(
                run_id=run_id,
                status="pending",
                episode_count=1,
                base_seed=100,
            )
            db.add(evaluation)
            db.commit()
            evaluation_id = evaluation.id

        entered = threading.Event()

        class SlowEnvironment:
            def reset(self, *, seed):
                return 0, {}

            def step(self, _action):
                entered.set()
                time.sleep(0.02)
                return 0, 1.0, False, False, {}

            def close(self):
                return None

        class FakeModel:
            def predict(self, *_args, **_kwargs):
                return 0, None

        artifact_dir = Path(self.temp_dir.name) / "models"
        artifact_dir.mkdir()
        (artifact_dir / f"run_{run_id}.zip").touch()
        with (
            patch.object(evaluation_service, "SessionLocal", self.session_factory),
            patch.object(evaluation_service.gym, "make", return_value=SlowEnvironment()),
            patch.object(evaluation_service.PPO, "load", return_value=FakeModel()),
            patch.object(evaluation_service, "MODEL_DIR", artifact_dir),
        ):
            evaluation_service.launch_evaluation(evaluation_id)
            self.assertTrue(entered.wait(2))
            evaluation_service.shutdown_evaluations(timeout=2)

        with self.session_factory() as db:
            evaluation = db.get(PolicyEvaluation, evaluation_id)
            episodes = db.query(EvaluationEpisode).filter_by(
                evaluation_id=evaluation_id
            ).all()
        self.assertEqual(evaluation.status, "stopped")
        self.assertIsNone(evaluation.mean_episode_reward)
        self.assertEqual(episodes, [])

    def test_training_launch_failure_marks_accepted_work_failed(self):
        with self.session_factory() as db:
            experiment = db.get(Experiment, self.experiment_id)
            experiment.status = "created"
            db.commit()
        from app.routers.experiments import start_experiment

        with (
            self.session_factory() as db,
            patch("app.routers.experiments.reserve_job") as router_reserve,
            patch(
                "app.routers.experiments.SessionLocal",
                self.session_factory,
            ),
            patch(
                "app.routers.experiments.launch_job",
                side_effect=RuntimeError("worker thread failed"),
            ),
        ):
            event = threading.Event()
            router_reserve.return_value = event
            with self.assertRaises(HTTPException) as error:
                start_experiment(self.experiment_id, db)
        self.assertEqual(error.exception.status_code, 503)

        with self.session_factory() as db:
            experiment = db.get(Experiment, self.experiment_id)
            runs = db.query(TrainingRun).filter_by(
                experiment_id=self.experiment_id
            ).all()
        self.assertEqual(experiment.status, "failed")
        self.assertTrue(runs)
        self.assertTrue(all(run.status == "failed" for run in runs))

    def test_training_failure_preserves_completed_episode_metrics(self):
        run_id = self._add_run(status="pending")
        with self.session_factory() as db:
            db.query(Experiment).filter_by(id=self.experiment_id).update(
                {Experiment.status: "pending"}
            )
            db.commit()

        class FakeEnvironment:
            def close(self):
                return None

        class FailingAfterMetricModel:
            def __init__(self, *_args, **_kwargs):
                self.num_timesteps = 12

            def learn(self, *, callback, **_kwargs):
                callback.model = SimpleNamespace()
                callback.num_timesteps = 12
                callback._on_training_start()
                callback.locals = {"infos": [{"episode": {"r": 8.5, "l": 12}}]}
                callback._on_step()
                raise RuntimeError("private training exception")

        with (
            patch.object(training_service, "SessionLocal", self.session_factory),
            patch.object(training_metrics, "SessionLocal", self.session_factory),
            patch.object(training_service.gym, "make", return_value=FakeEnvironment()),
            patch.object(training_service, "PPO", FailingAfterMetricModel),
            patch.object(training_service, "MODEL_DIR", Path(self.temp_dir.name)),
        ):
            training_service.run_experiment(self.experiment_id, threading.Event())

        with self.session_factory() as db:
            run = db.get(TrainingRun, run_id)
            experiment = db.get(Experiment, self.experiment_id)
            metrics = db.query(TrainingMetric).filter_by(run_id=run_id).all()
        self.assertEqual(run.status, "failed")
        self.assertEqual(experiment.status, "failed")
        self.assertTrue(any(metric.episode_reward == 8.5 for metric in metrics))
        self.assertNotIn("private training exception", run.failure_message)


if __name__ == "__main__":
    unittest.main()
