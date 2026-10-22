import asyncio
import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, WebSocketDisconnect
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Experiment, TrainingMetric
from app.routers.experiments import (
    experiment_metrics_websocket,
    get_experiment_metrics,
)
from app.services import training_service
from app.services.metric_stream import ExperimentMetricStream, metric_stream
from app.services.training_metrics import TrainingMetricsCallback
from app.schemas.metrics import TrainingMetricResponse


class TrainingMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test_metrics.db"
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
        with self.session_factory() as db:
            experiment = Experiment(
                name="Metric run",
                environment="InvertedPendulum-v5",
                algorithm="PPO",
                total_timesteps=100,
                status="created",
            )
            db.add(experiment)
            db.commit()
            self.experiment_id = experiment.id

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def add_metric(
        self,
        timestep,
        timestamp,
        reward=None,
        length=None,
        status="running",
        experiment_id=None,
    ):
        with self.session_factory() as db:
            metric = TrainingMetric(
                experiment_id=experiment_id or self.experiment_id,
                global_timestep=timestep,
                completed_episode_count=1 if reward is not None else 0,
                episode_reward=reward,
                episode_length=length,
                recent_mean_episode_reward=reward,
                elapsed_seconds=1.0,
                timesteps_per_second=float(timestep),
                training_status=status,
                timestamp=timestamp,
            )
            db.add(metric)
            db.commit()
            db.refresh(metric)
            return metric.id

    def test_metrics_table_is_created_with_foreign_key(self):
        table_names = inspect(self.engine).get_table_names()
        self.assertIn("training_metrics", table_names)
        foreign_keys = inspect(self.engine).get_foreign_keys("training_metrics")
        self.assertEqual(foreign_keys[0]["referred_table"], "experiments")

    def test_history_orders_metrics_and_applies_limit_and_timestep_range(self):
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.add_metric(10, base_time, reward=3.0, length=5)
        self.add_metric(10, base_time + timedelta(seconds=1), reward=4.0, length=7)
        self.add_metric(20, base_time + timedelta(seconds=2), reward=5.0, length=8)

        with self.session_factory() as db:
            ordered = get_experiment_metrics(
                self.experiment_id,
                3,
                None,
                None,
                db,
            )
            limited = get_experiment_metrics(
                self.experiment_id,
                2,
                10,
                20,
                db,
            )

        self.assertEqual([metric.global_timestep for metric in ordered], [10, 10, 20])
        self.assertEqual([metric.episode_reward for metric in ordered], [3.0, 4.0, 5.0])
        self.assertEqual([metric.global_timestep for metric in limited], [10, 20])
        self.assertEqual([metric.episode_reward for metric in limited], [4.0, 5.0])

    def test_history_rejects_missing_experiment_and_invalid_range(self):
        with self.session_factory() as db:
            with self.assertRaises(HTTPException) as missing:
                get_experiment_metrics(999, 100, None, None, db)
            self.assertEqual(missing.exception.status_code, 404)

            with self.assertRaises(HTTPException) as invalid_range:
                get_experiment_metrics(self.experiment_id, 100, 20, 10, db)
            self.assertEqual(invalid_range.exception.status_code, 422)

    def test_callback_persists_real_episode_metrics(self):
        cancel_event = threading.Event()
        with patch(
            "app.services.training_metrics.SessionLocal",
            self.session_factory,
        ):
            callback = TrainingMetricsCallback(
                self.experiment_id,
                cancel_event,
                log_interval=50,
            )
            callback.model = SimpleNamespace()
            callback.num_timesteps = 12
            callback.locals = {
                "infos": [{"episode": {"r": 3.5, "l": 12}}],
            }

            self.assertTrue(callback._on_step())
            callback.finalize("completed")

        with self.session_factory() as db:
            metrics = (
                db.query(TrainingMetric)
                .filter_by(experiment_id=self.experiment_id)
                .order_by(TrainingMetric.id)
                .all()
            )
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].global_timestep, 12)
        self.assertEqual(metrics[0].completed_episode_count, 1)
        self.assertEqual(metrics[0].episode_reward, 3.5)
        self.assertEqual(metrics[0].episode_length, 12)
        self.assertEqual(metrics[0].recent_mean_episode_reward, 3.5)
        self.assertGreaterEqual(metrics[0].elapsed_seconds, 0)
        self.assertGreaterEqual(metrics[0].timesteps_per_second, 0)
        payload = TrainingMetricResponse.model_validate(metrics[0]).model_dump(
            mode="json"
        )
        self.assertTrue(payload["timestamp"].endswith("+00:00"))
        self.assertEqual(metrics[1].training_status, "completed")

    def test_callback_does_not_invent_episode_data(self):
        with patch(
            "app.services.training_metrics.SessionLocal",
            self.session_factory,
        ):
            callback = TrainingMetricsCallback(
                self.experiment_id,
                threading.Event(),
                log_interval=10,
            )
            callback.model = SimpleNamespace()
            callback.num_timesteps = 1
            callback.locals = {"infos": [{}]}
            self.assertTrue(callback._on_step())

            with self.session_factory() as db:
                self.assertEqual(db.query(TrainingMetric).count(), 0)

            callback.num_timesteps = 10
            self.assertTrue(callback._on_step())

        with self.session_factory() as db:
            metric = db.query(TrainingMetric).one()
        self.assertIsNone(metric.episode_reward)
        self.assertIsNone(metric.episode_length)
        self.assertIsNone(metric.recent_mean_episode_reward)
        self.assertEqual(metric.completed_episode_count, 0)

    def test_broker_bounds_queues_and_isolates_experiments(self):
        stream = ExperimentMetricStream(queue_size=1, max_subscribers_per_experiment=1)
        first = stream.subscribe(self.experiment_id)
        second_experiment = self.experiment_id + 1
        second = stream.subscribe(second_experiment)
        stream.publish(self.experiment_id, {"value": 1})
        stream.publish(self.experiment_id, {"value": 2})

        self.assertEqual(first.qsize(), 1)
        self.assertEqual(first.get_nowait(), {"type": "resync_required"})
        self.assertTrue(second.empty())
        with self.assertRaises(RuntimeError):
            stream.subscribe(self.experiment_id)

        stream.unsubscribe(self.experiment_id, first)
        stream.unsubscribe(second_experiment, second)
        self.assertEqual(stream.subscriber_count(self.experiment_id), 0)
        self.assertEqual(stream.subscriber_count(second_experiment), 0)

    def test_websocket_sends_metrics_and_cleans_up_disconnected_client(self):
        experiment_id = self.experiment_id
        metric_stream.publish(
            experiment_id,
            {"type": "metric", "metric": {"id": 500, "global_timestep": 8}},
        )

        class FakeWebSocket:
            def __init__(self):
                self.messages = []

            async def accept(self):
                return None

            async def receive_text(self):
                await asyncio.Future()

            async def send_json(self, message):
                self.messages.append(message)
                if message["type"] == "status":
                    metric_stream.publish(
                        experiment_id,
                        {
                            "type": "metric",
                            "metric": {"id": 501, "global_timestep": 9},
                        },
                    )
                else:
                    raise WebSocketDisconnect(code=1000)

            async def close(self, code):
                self.close_code = code

        websocket = FakeWebSocket()
        with self.session_factory() as db:
            asyncio.run(
                experiment_metrics_websocket(
                    websocket,
                    self.experiment_id,
                    db,
                )
            )

        self.assertEqual(websocket.messages[0]["type"], "status")
        self.assertEqual(websocket.messages[0]["experiment_id"], self.experiment_id)
        self.assertEqual(websocket.messages[1]["metric"]["id"], 501)
        self.assertEqual(metric_stream.subscriber_count(self.experiment_id), 0)

    def test_websocket_asgi_connection_streams_experiment_metric(self):
        experiment_id = self.experiment_id
        messages = []
        disconnect_ready = asyncio.Event()
        connection_requested = False

        def isolated_database():
            with self.session_factory() as db:
                yield db

        previous_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_db] = isolated_database

        async def receive():
            nonlocal connection_requested
            if not connection_requested:
                connection_requested = True
                return {"type": "websocket.connect"}
            await disconnect_ready.wait()
            return {"type": "websocket.disconnect", "code": 1000}

        async def send(message):
            messages.append(message)
            if message["type"] != "websocket.send":
                return
            event = json.loads(message["text"])
            if event["type"] == "status":
                metric_stream.publish(
                    experiment_id,
                    {
                        "type": "metric",
                        "metric": {
                            "id": 700,
                            "experiment_id": experiment_id,
                            "global_timestep": 15,
                        },
                    },
                )
            elif event["type"] == "metric":
                disconnect_ready.set()

        path = f"/experiments/{experiment_id}/metrics/ws"
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "root_path": "",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"origin", b"http://localhost:5173"),
            ],
            "subprotocols": [],
            "state": {},
        }
        try:
            asyncio.run(app(scope, receive, send))
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(previous_overrides)

        event_payloads = [
            json.loads(message["text"])
            for message in messages
            if message["type"] == "websocket.send"
        ]
        self.assertEqual(messages[0]["type"], "websocket.accept")
        self.assertEqual(event_payloads[0]["type"], "status")
        self.assertEqual(event_payloads[1]["metric"]["experiment_id"], experiment_id)
        self.assertEqual(metric_stream.subscriber_count(experiment_id), 0)

    def test_websocket_initial_message_includes_persisted_metrics(self):
        self.add_metric(
            20,
            datetime.now(timezone.utc),
            reward=7.0,
            length=20,
            status="completed",
        )
        with self.session_factory() as db:
            db.query(Experiment).filter_by(id=self.experiment_id).update(
                {Experiment.status: "completed"}
            )
            db.commit()

        class FakeWebSocket:
            async def accept(self):
                return None

            async def send_json(self, message):
                self.message = message

            async def close(self, code):
                self.close_code = code

        websocket = FakeWebSocket()
        with self.session_factory() as db:
            asyncio.run(
                experiment_metrics_websocket(
                    websocket,
                    self.experiment_id,
                    db,
                )
            )
        self.assertEqual(websocket.message["status"], "completed")
        self.assertEqual(websocket.message["latest_metric"]["episode_reward"], 7.0)
        self.assertEqual(websocket.close_code, 1000)
        self.assertEqual(metric_stream.subscriber_count(self.experiment_id), 0)

    def test_training_completion_persists_terminal_status(self):
        class FakeEnvironment:
            observation_space = None
            action_space = None

            def close(self):
                return None

        class CompletingModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def learn(self, *, total_timesteps, callback):
                callback.model = SimpleNamespace()
                callback.num_timesteps = total_timesteps
                callback._on_training_start()
                callback.locals = {
                    "infos": [{"episode": {"r": 9.0, "l": total_timesteps}}],
                }
                callback._on_step()

            def save(self, _path):
                return None

        with (
            patch.object(training_service, "SessionLocal", self.session_factory),
            patch(
                "app.services.training_metrics.SessionLocal",
                self.session_factory,
            ),
            patch.object(training_service.gym, "make", return_value=FakeEnvironment()),
            patch.object(training_service, "PPO", CompletingModel),
            patch.object(training_service, "MODEL_DIR", Path(self.temp_dir.name)),
        ):
            with self.session_factory() as db:
                db.query(Experiment).filter_by(id=self.experiment_id).update(
                    {Experiment.status: "pending"}
                )
                db.commit()
            training_service.run_experiment(self.experiment_id, threading.Event())

        with self.session_factory() as db:
            experiment = db.get(Experiment, self.experiment_id)
            metrics = (
                db.query(TrainingMetric)
                .filter_by(experiment_id=self.experiment_id)
                .order_by(TrainingMetric.id)
                .all()
            )
        self.assertEqual(experiment.status, "completed")
        self.assertEqual(
            next(metric for metric in metrics if metric.episode_reward is not None)
            .episode_reward,
            9.0,
        )
        self.assertEqual(metrics[-1].training_status, "completed")


if __name__ == "__main__":
    unittest.main()
