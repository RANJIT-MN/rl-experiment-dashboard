import asyncio
import csv
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Experiment, TrainingMetric
from app.routers import comparison
from app.schemas.comparison import ExperimentComparisonRequest
from pydantic import ValidationError


class ExperimentComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "comparison.db"
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
            self.first_id = self._add_experiment(
                db,
                'Run, "Alpha"\nsecond line',
                learning_rate=0.001,
                n_steps=512,
                batch_size=32,
                gamma=0.95,
                n_epochs=4,
            )
            self.second_id = self._add_experiment(
                db,
                "Beta",
                learning_rate=None,
                n_steps=None,
                batch_size=None,
                gamma=None,
                n_epochs=None,
            )
            self.unselected_id = self._add_experiment(db, "Unselected")
            self._add_metric(
                db,
                self.first_id,
                10,
                3.0,
                5,
                100.0,
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            self._add_metric(
                db,
                self.first_id,
                20,
                -1.0,
                7,
                110.0,
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1),
            )
            self._add_metric(
                db,
                self.first_id,
                30,
                0.0,
                9,
                120.0,
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=2),
            )
            self._add_metric(
                db,
                self.second_id,
                20,
                10.0,
                2,
                50.0,
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
            self._add_metric(
                db,
                self.unselected_id,
                99,
                999.0,
                999,
                999.0,
                datetime(2026, 1, 3, tzinfo=timezone.utc),
            )
            db.query(Experiment).filter_by(id=self.second_id).update(
                {
                    Experiment.learning_rate: None,
                    Experiment.n_steps: None,
                    Experiment.batch_size: None,
                    Experiment.gamma: None,
                    Experiment.n_epochs: None,
                }
            )
            db.commit()

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    @staticmethod
    def _add_experiment(db, name, **configuration):
        experiment = Experiment(
            name=name,
            environment="InvertedPendulum-v5",
            algorithm="PPO",
            total_timesteps=1000,
            status="completed",
            **configuration,
        )
        db.add(experiment)
        db.flush()
        return experiment.id

    @staticmethod
    def _add_metric(db, experiment_id, timestep, reward, length, speed, timestamp):
        db.add(
            TrainingMetric(
                experiment_id=experiment_id,
                global_timestep=timestep,
                completed_episode_count=timestep // 10,
                episode_reward=reward,
                episode_length=length,
                recent_mean_episode_reward=reward,
                elapsed_seconds=1.0,
                timesteps_per_second=speed,
                training_status="completed",
                timestamp=timestamp,
            )
        )
        db.flush()

    def test_comparison_selects_only_requested_experiments_and_summarizes_metrics(self):
        request = ExperimentComparisonRequest(
            experiment_ids=[self.first_id, self.second_id]
        )
        with self.session_factory() as db:
            result = comparison.compare_experiments(request, db)

        self.assertEqual(
            [item.experiment.id for item in result.experiments],
            [self.first_id, self.second_id],
        )
        first, second = result.experiments
        self.assertEqual(first.completed_episode_count, 3)
        self.assertEqual(first.latest_episode_reward, 0.0)
        self.assertAlmostEqual(first.mean_episode_reward, 2 / 3)
        self.assertEqual(first.maximum_episode_reward, 3.0)
        self.assertEqual(first.minimum_episode_reward, -1.0)
        self.assertEqual(first.mean_episode_length, 7.0)
        self.assertEqual(first.latest_training_speed, 120.0)
        self.assertEqual(
            [metric.experiment_id for metric in first.curve_metrics],
            [self.first_id] * 3,
        )
        self.assertEqual(
            [metric.episode_reward for metric in first.curve_metrics],
            [3.0, -1.0, 0.0],
        )
        self.assertEqual(second.completed_episode_count, 1)
        self.assertEqual(second.mean_episode_reward, 10.0)
        self.assertIn("every persisted completed-episode sample", result.metric_definition)

    def test_comparison_with_no_metrics_returns_missing_values(self):
        with self.session_factory() as db:
            empty_id = self._add_experiment(db, "Empty run")
            request = ExperimentComparisonRequest(
                experiment_ids=[empty_id, self.second_id]
            )
            item = comparison.compare_experiments(request, db).experiments[0]

        self.assertEqual(item.completed_episode_count, 0)
        self.assertIsNone(item.latest_episode_reward)
        self.assertIsNone(item.mean_episode_reward)
        self.assertIsNone(item.maximum_episode_reward)
        self.assertIsNone(item.minimum_episode_reward)
        self.assertIsNone(item.mean_episode_length)
        self.assertIsNone(item.latest_training_speed)
        self.assertEqual(item.curve_metrics, [])

    def test_comparison_rejects_nonexistent_experiment_ids(self):
        request = ExperimentComparisonRequest(
            experiment_ids=[self.first_id, 9999]
        )
        with self.session_factory() as db:
            with self.assertRaises(HTTPException) as error:
                comparison.compare_experiments(request, db)
        self.assertEqual(error.exception.status_code, 404)
        self.assertIn("9999", error.exception.detail)

    def test_comparison_api_returns_only_requested_experiments(self):
        request_body = json.dumps(
            {"experiment_ids": [self.first_id, self.second_id]}
        ).encode()
        response_messages = []
        received_request = False

        def isolated_database():
            with self.session_factory() as db:
                yield db

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

        path = "/experiments/compare"
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
        result = json.loads(response_body)
        self.assertEqual(response_start["status"], 200)
        self.assertEqual(
            [item["experiment"]["id"] for item in result["experiments"]],
            [self.first_id, self.second_id],
        )

    def test_comparison_request_rejects_duplicate_and_invalid_ids(self):
        for ids in (
            [self.first_id, self.first_id],
            [self.first_id, 0],
            [self.first_id, -1],
            [self.first_id, "2"],
            [self.first_id],
            list(range(1, 12)),
        ):
            with self.subTest(ids=ids), self.assertRaises(ValidationError):
                ExperimentComparisonRequest.model_validate({"experiment_ids": ids})

    def test_legacy_configuration_is_reported_as_unrecorded(self):
        request = ExperimentComparisonRequest(
            experiment_ids=[self.first_id, self.second_id]
        )
        with self.session_factory() as db:
            result = comparison.compare_experiments(request, db)

        legacy = result.experiments[1].experiment
        self.assertIsNone(legacy.learning_rate)
        self.assertIsNone(legacy.n_steps)
        self.assertIsNone(legacy.batch_size)
        self.assertIsNone(legacy.gamma)
        self.assertIsNone(legacy.n_epochs)

    def test_chart_limit_is_bounded_and_experiment_points_stay_associated(self):
        request = ExperimentComparisonRequest(
            experiment_ids=[self.first_id, self.second_id]
        )
        with (
            self.session_factory() as db,
            patch.object(comparison, "MAX_CURVE_POINTS", 1),
        ):
            result = comparison.compare_experiments(request, db)

        self.assertTrue(result.experiments[0].curve_truncated)
        self.assertEqual(len(result.experiments[0].curve_metrics), 1)
        self.assertEqual(
            result.experiments[0].curve_metrics[0].experiment_id,
            self.first_id,
        )
        self.assertFalse(result.experiments[1].curve_truncated)
        self.assertEqual(
            result.experiments[1].curve_metrics[0].experiment_id,
            self.second_id,
        )

    def test_csv_stream_has_escaped_headers_and_only_selected_records(self):
        request = ExperimentComparisonRequest(
            experiment_ids=[self.first_id, self.second_id]
        )
        with (
            self.session_factory() as db,
            patch.object(comparison, "SessionLocal", self.session_factory),
        ):
            response = comparison.export_comparison_metrics(request, db)
            chunks = asyncio.run(self._read_stream(response))

        content = b"".join(chunks)
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        self.assertEqual(rows[0], comparison.CSV_HEADERS)
        self.assertEqual(len(rows), 5)
        self.assertEqual({row[0] for row in rows[1:]}, {str(self.first_id), str(self.second_id)})
        self.assertEqual(
            {row[0] for row in rows[1:]},
            {str(self.first_id), str(self.second_id)},
        )
        self.assertEqual(rows[1][1], 'Run, "Alpha"\nsecond line')
        self.assertEqual(rows[1][10:15], ["0.001", "512", "32", "0.95", "4"])
        self.assertEqual(rows[4][10:15], ["", "", "", "", ""])
        self.assertIn("filename=\"experiment-comparison-", response.headers["content-disposition"])

    @staticmethod
    async def _read_stream(response):
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
        return chunks


if __name__ == "__main__":
    unittest.main()
