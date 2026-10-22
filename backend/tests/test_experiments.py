import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import database
from app.database import Base
from app.database import get_db
from app.main import app
from app.models import Experiment
from app.routers.experiments import (
    create_experiment,
    list_experiments,
    start_experiment,
    stop_experiment,
)
from app.schemas import ExperimentCreate, ExperimentResponse
from app.services import training_service


class ExperimentEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test_experiments.db"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def tearDown(self):
        for experiment_id in range(1, 20):
            training_service.release_job(experiment_id)
        self.engine.dispose()
        self.temp_dir.cleanup()

    def add_experiment(self, status="created"):
        with self.session_factory() as session:
            experiment = Experiment(
                name="Lifecycle run",
                environment="InvertedPendulum-v5",
                algorithm="PPO",
                total_timesteps=8,
                status=status,
            )
            session.add(experiment)
            session.commit()
            return experiment.id

    def test_create_experiment_persists_record(self):
        payload = ExperimentCreate(name="PPO run", total_timesteps=250)
        with self.session_factory() as session:
            created = create_experiment(payload, session)

        self.assertIsNotNone(created.id)
        self.assertEqual(created.name, "PPO run")
        self.assertEqual(created.environment, "InvertedPendulum-v5")

        with self.session_factory() as session:
            persisted = session.get(Experiment, created.id)
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted.total_timesteps, 250)

    def test_create_experiment_persists_sb3_default_parameters(self):
        payload = ExperimentCreate(name="Default PPO config")
        with self.session_factory() as session:
            created = create_experiment(payload, session)
            persisted = session.get(Experiment, created.id)

        self.assertEqual(created.learning_rate, 3e-4)
        self.assertEqual(created.n_steps, 2048)
        self.assertEqual(created.batch_size, 64)
        self.assertEqual(created.gamma, 0.99)
        self.assertEqual(created.n_epochs, 10)
        self.assertEqual(persisted.n_steps, 2048)

    def test_create_and_list_experiments_preserve_custom_parameters(self):
        payload = ExperimentCreate(
            name="Custom PPO config",
            total_timesteps=4096,
            learning_rate=0.001,
            n_steps=512,
            batch_size=32,
            gamma=0.95,
            n_epochs=4,
        )
        with self.session_factory() as session:
            created = create_experiment(payload, session)
            listed = list_experiments(session)

        saved = next(experiment for experiment in listed if experiment.id == created.id)
        self.assertEqual(
            (
                saved.learning_rate,
                saved.n_steps,
                saved.batch_size,
                saved.gamma,
                saved.n_epochs,
            ),
            (0.001, 512, 32, 0.95, 4),
        )
        response = ExperimentResponse.model_validate(saved)
        self.assertEqual(response.n_steps, 512)

    def test_list_experiments_returns_saved_records(self):
        with self.session_factory() as session:
            create_experiment(ExperimentCreate(name="History entry"), session)

        with self.session_factory() as session:
            results = list_experiments(session)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "History entry")

    def test_validation_rejects_blank_name_and_invalid_timesteps(self):
        for payload in (
            {"name": "   ", "total_timesteps": 10},
            {"name": "Invalid run", "total_timesteps": 0},
            {"name": "Invalid run", "total_timesteps": -1},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    ExperimentCreate.model_validate(payload)

    def test_validation_rejects_unsupported_environment_and_algorithm(self):
        for payload in (
            {"name": "Invalid env", "environment": "CartPole-v1"},
            {"name": "Invalid algorithm", "algorithm": "SAC"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    ExperimentCreate.model_validate(payload)

    def test_default_timestep_count_is_positive(self):
        payload = ExperimentCreate(name="Default run")
        self.assertEqual(payload.total_timesteps, 1000)

    def test_validation_rejects_invalid_ppo_hyperparameters(self):
        invalid_payloads = (
            {"learning_rate": 0},
            {"learning_rate": -0.1},
            {"learning_rate": float("nan")},
            {"learning_rate": float("inf")},
            {"gamma": 0},
            {"gamma": -0.1},
            {"gamma": 1.01},
            {"gamma": float("nan")},
            {"gamma": float("inf")},
            {"n_steps": 0},
            {"n_steps": -2},
            {"n_steps": 1},
            {"n_steps": 128.5},
            {"batch_size": 0},
            {"batch_size": -2},
            {"batch_size": 1},
            {"batch_size": 3},
            {"batch_size": 2049},
            {"batch_size": 32.5},
            {"n_epochs": 0},
            {"n_epochs": -1},
            {"n_epochs": 2.5},
            {"n_steps": 96, "batch_size": 64},
            {"n_steps": 32, "batch_size": 64},
        )
        for override in invalid_payloads:
            with self.subTest(override=override):
                payload = {"name": "Invalid PPO config", **override}
                with self.assertRaises(ValidationError):
                    ExperimentCreate.model_validate(payload)

    def test_invalid_hyperparameters_return_http_422(self):
        request_body = json.dumps(
            {
                "name": "Invalid gamma",
                "gamma": 0,
            }
        ).encode()
        response_messages = []
        received_request = False

        def isolated_database():
            with self.session_factory() as session:
                yield session

        previous_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_db] = isolated_database

        async def receive():
            nonlocal received_request
            if not received_request:
                received_request = True
                return {
                    "type": "http.request",
                    "body": request_body,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message):
            response_messages.append(message)

        path = "/experiments"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(request_body)).encode()),
            ],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
            "state": {},
        }
        try:
            asyncio.run(app(scope, receive, send))
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(previous_overrides)

        response_start = next(
            message
            for message in response_messages
            if message["type"] == "http.response.start"
        )
        response_body = next(
            message["body"]
            for message in response_messages
            if message["type"] == "http.response.body"
        )
        self.assertEqual(response_start["status"], 422)
        self.assertIn(b"gamma", response_body)

    def test_legacy_schema_gets_additive_failure_message_column(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        legacy_engine = create_engine(f"sqlite:///{legacy_path.as_posix()}")
        with legacy_engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE experiments ("
                "id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, "
                "environment VARCHAR NOT NULL, algorithm VARCHAR NOT NULL, "
                "status VARCHAR NOT NULL, total_timesteps INTEGER NOT NULL, "
                "created_at DATETIME NOT NULL, completed_at DATETIME)"
            )
            connection.exec_driver_sql(
                "INSERT INTO experiments "
                "(id,name,environment,algorithm,status,total_timesteps,created_at) "
                "VALUES (7,'Preserved','InvertedPendulum-v5','PPO','created',12,'2026-01-01')"
            )

        with patch.object(database, "engine", legacy_engine):
            database.initialize_database()

        with legacy_engine.connect() as connection:
            inspector = __import__("sqlalchemy").inspect(legacy_engine)
            columns = {column["name"] for column in inspector.get_columns("experiments")}
            tables = inspector.get_table_names()
            preserved = connection.exec_driver_sql(
                "SELECT id,name FROM experiments WHERE id=7"
            ).one()
        self.assertIn("failure_message", columns)
        self.assertIn("training_metrics", tables)
        self.assertEqual(preserved, (7, "Preserved"))
        self.assertTrue(
            {"learning_rate", "n_steps", "batch_size", "gamma", "n_epochs"}
            .issubset(columns)
        )
        with legacy_engine.connect() as connection:
            legacy_configuration = connection.exec_driver_sql(
                "SELECT learning_rate,n_steps,batch_size,gamma,n_epochs "
                "FROM experiments WHERE id=7"
            ).one()
        self.assertEqual(legacy_configuration, (None, None, None, None, None))

        legacy_sessions = sessionmaker(bind=legacy_engine)
        with legacy_sessions() as session:
            legacy_experiment = session.get(Experiment, 7)
            response = ExperimentResponse.model_validate(legacy_experiment)
        self.assertIsNone(response.learning_rate)
        self.assertIsNone(response.n_steps)
        self.assertIsNone(response.batch_size)
        self.assertIsNone(response.gamma)
        self.assertIsNone(response.n_epochs)
        legacy_engine.dispose()

    def test_training_worker_uses_saved_ppo_configuration(self):
        experiment_id = self.add_experiment(status="pending")
        with self.session_factory() as session:
            experiment = session.get(Experiment, experiment_id)
            experiment.learning_rate = 0.001
            experiment.n_steps = 512
            experiment.batch_size = 32
            experiment.gamma = 0.95
            experiment.n_epochs = 4
            session.commit()

        class FakeEnvironment:
            observation_space = None
            action_space = None

            def close(self):
                return None

        captured_configuration = {}

        class FakeModel:
            def __init__(self, *_args, **kwargs):
                captured_configuration.update(kwargs)

            def learn(self, **_kwargs):
                return None

            def save(self, _path):
                return None

        with (
            patch.object(training_service, "SessionLocal", self.session_factory),
            patch.object(training_service.gym, "make", return_value=FakeEnvironment()),
            patch.object(training_service, "PPO", FakeModel),
            patch.object(training_service, "MODEL_DIR", Path(self.temp_dir.name)),
        ):
            training_service.run_experiment(experiment_id, threading.Event())

        config = captured_configuration
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertEqual(config["n_steps"], 512)
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["gamma"], 0.95)
        self.assertEqual(config["n_epochs"], 4)

    def test_legacy_experiment_cannot_restart_without_saved_configuration(self):
        legacy_path = Path(self.temp_dir.name) / "legacy_restart.db"
        legacy_engine = create_engine(f"sqlite:///{legacy_path.as_posix()}")
        with legacy_engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE experiments ("
                "id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, "
                "environment VARCHAR NOT NULL, algorithm VARCHAR NOT NULL, "
                "status VARCHAR NOT NULL, total_timesteps INTEGER NOT NULL, "
                "created_at DATETIME NOT NULL, completed_at DATETIME)"
            )
            connection.exec_driver_sql(
                "INSERT INTO experiments "
                "(id,name,environment,algorithm,status,total_timesteps,created_at) "
                "VALUES (1,'Legacy','InvertedPendulum-v5','PPO','created',1000,'2026-01-01')"
            )
        with patch.object(database, "engine", legacy_engine):
            database.initialize_database()
        legacy_sessions = sessionmaker(bind=legacy_engine)
        with legacy_sessions() as session:
            with self.assertRaises(HTTPException) as error:
                start_experiment(1, session)
        self.assertEqual(error.exception.status_code, 409)
        self.assertIn("no saved PPO configuration", error.exception.detail)
        legacy_engine.dispose()

    def test_start_valid_experiment_claims_pending_job(self):
        experiment_id = self.add_experiment()
        with self.session_factory() as session, patch(
            "app.routers.experiments.launch_job"
        ) as launch:
            result = start_experiment(experiment_id, session)

        self.assertEqual(result.status, "pending")
        self.assertEqual(result.message, "Training accepted and queued.")
        launch.assert_called_once()

    def test_start_rejects_missing_experiment(self):
        with self.session_factory() as session:
            with self.assertRaises(HTTPException) as error:
                start_experiment(999, session)
        self.assertEqual(error.exception.status_code, 404)

    def test_start_rejects_duplicate_start(self):
        experiment_id = self.add_experiment()
        with self.session_factory() as session, patch(
            "app.routers.experiments.launch_job"
        ):
            start_experiment(experiment_id, session)

        with self.session_factory() as session:
            with self.assertRaises(HTTPException) as error:
                start_experiment(experiment_id, session)
        self.assertEqual(error.exception.status_code, 409)

    def test_start_rejects_completed_state(self):
        experiment_id = self.add_experiment(status="completed")
        with self.session_factory() as session:
            with self.assertRaises(HTTPException) as error:
                start_experiment(experiment_id, session)
        self.assertEqual(error.exception.status_code, 409)

    def test_stop_requests_cancellation_without_claiming_stopped(self):
        experiment_id = self.add_experiment(status="running")
        event = training_service.reserve_job(experiment_id)
        with self.session_factory() as session:
            result = stop_experiment(experiment_id, session)
            persisted = session.get(Experiment, experiment_id)

        self.assertTrue(event.is_set())
        self.assertEqual(result.status, "stopping")
        self.assertEqual(persisted.status, "stopping")

    def test_worker_records_stopped_only_after_learn_exits(self):
        experiment_id = self.add_experiment(status="pending")
        cancel_event = threading.Event()
        learn_entered = threading.Event()
        allow_learn_exit = threading.Event()
        environment_closed = threading.Event()

        class FakeEnvironment:
            observation_space = None
            action_space = None

            def close(self):
                environment_closed.set()

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def learn(self, *, total_timesteps, callback):
                learn_entered.set()
                while not cancel_event.is_set():
                    cancel_event.wait(0.01)
                self.acknowledged = not callback._on_step()
                allow_learn_exit.wait(2)

            def save(self, _path):
                raise AssertionError("Cancelled training must not save a completed model")

        with (
            patch.object(training_service, "SessionLocal", self.session_factory),
            patch.object(training_service.gym, "make", return_value=FakeEnvironment()),
            patch.object(training_service, "PPO", FakeModel),
        ):
            worker = threading.Thread(
                target=training_service.run_experiment,
                args=(experiment_id, cancel_event),
            )
            worker.start()
            self.assertTrue(learn_entered.wait(2))
            with self.session_factory() as session:
                session.query(Experiment).filter_by(id=experiment_id).update(
                    {Experiment.status: "stopping"}
                )
                session.commit()
            cancel_event.set()

            with self.session_factory() as session:
                self.assertEqual(session.get(Experiment, experiment_id).status, "stopping")

            allow_learn_exit.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(environment_closed.is_set())
        with self.session_factory() as session:
            stopped = session.get(Experiment, experiment_id)
            self.assertEqual(stopped.status, "stopped")
            self.assertIsNotNone(stopped.completed_at)

    def test_training_failure_records_failed_and_closes_environment(self):
        experiment_id = self.add_experiment(status="pending")
        environment_closed = threading.Event()

        class FakeEnvironment:
            observation_space = None
            action_space = None

            def close(self):
                environment_closed.set()

        class FailingModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def learn(self, **_kwargs):
                raise RuntimeError("private detail that must not be returned")

        with (
            patch.object(training_service, "SessionLocal", self.session_factory),
            patch.object(training_service.gym, "make", return_value=FakeEnvironment()),
            patch.object(training_service, "PPO", FailingModel),
        ):
            training_service.run_experiment(experiment_id, threading.Event())

        self.assertTrue(environment_closed.is_set())
        with self.session_factory() as session:
            failed = session.get(Experiment, experiment_id)
            self.assertEqual(failed.status, "failed")
            self.assertIn("RuntimeError", failed.failure_message)
            self.assertNotIn("private detail", failed.failure_message)
            self.assertIsNotNone(failed.completed_at)


if __name__ == "__main__":
    unittest.main()