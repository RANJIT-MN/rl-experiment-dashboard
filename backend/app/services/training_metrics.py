import math
import os
import threading
import time
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Real

from stable_baselines3.common.callbacks import BaseCallback

from app.database import SessionLocal
from app.models import TrainingMetric
from app.schemas.metrics import TrainingMetricResponse
from app.services.metric_stream import metric_stream


DEFAULT_METRIC_LOG_INTERVAL = 1000
RECENT_REWARD_WINDOW = 100


def get_metric_log_interval() -> int:
    value = os.getenv("RL_METRICS_LOG_INTERVAL", str(DEFAULT_METRIC_LOG_INTERVAL))
    try:
        interval = int(value)
    except ValueError as error:
        raise ValueError("RL_METRICS_LOG_INTERVAL must be a positive integer") from error
    if interval < 1:
        raise ValueError("RL_METRICS_LOG_INTERVAL must be a positive integer")
    return interval


class CancellationCallback(BaseCallback):
    def __init__(self, cancel_event: threading.Event):
        super().__init__()
        self.cancel_event = cancel_event
        self.cancelled = False

    def _on_step(self) -> bool:
        self.cancelled = self.cancel_event.is_set()
        return not self.cancelled


class TrainingMetricsCallback(CancellationCallback):
    def __init__(
        self,
        experiment_id: int,
        cancel_event: threading.Event,
        log_interval: int | None = None,
        run_id: int | None = None,
    ):
        super().__init__(cancel_event)
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.log_interval = (
            get_metric_log_interval() if log_interval is None else log_interval
        )
        if self.log_interval < 1:
            raise ValueError("Metric logging interval must be a positive integer")
        self.started_monotonic = time.monotonic()
        self.completed_episode_count = 0
        self.recent_rewards: deque[float] = deque(maxlen=RECENT_REWARD_WINDOW)
        self.last_logged_timestep = 0
        self.current_timestep = 0

    def _on_training_start(self) -> None:
        self._persist_metric(
            status="stopping" if self.cancel_event.is_set() else "running",
            episode_reward=None,
            episode_length=None,
        )

    def _on_step(self) -> bool:
        if getattr(self, "model", None) is None:
            return super()._on_step()

        self.current_timestep = self.num_timesteps
        completed_episodes = self._completed_episodes()

        if completed_episodes:
            self.completed_episode_count += len(completed_episodes)
            for reward, _ in completed_episodes:
                self.recent_rewards.append(reward)

        interval_elapsed = (
            self.current_timestep - self.last_logged_timestep >= self.log_interval
        )
        if completed_episodes or interval_elapsed:
            latest_reward, latest_length = (
                completed_episodes[-1] if completed_episodes else (None, None)
            )
            self._persist_metric(
                status="stopping" if self.cancel_event.is_set() else "running",
                episode_reward=latest_reward,
                episode_length=latest_length,
            )
            self.last_logged_timestep = self.current_timestep

        return super()._on_step()

    def _completed_episodes(self) -> list[tuple[float, int]]:
        episodes = []
        for info in self.locals.get("infos", ()):
            episode = info.get("episode") if isinstance(info, Mapping) else None
            if not isinstance(episode, Mapping):
                continue
            reward = episode.get("r")
            length = episode.get("l")
            if not isinstance(reward, Real) or not math.isfinite(float(reward)):
                continue
            if (
                not isinstance(length, Real)
                or not math.isfinite(float(length))
                or length < 0
                or not float(length).is_integer()
            ):
                continue
            episodes.append((float(reward), int(length)))
        return episodes

    def _persist_metric(
        self,
        status: str,
        episode_reward: float | None,
        episode_length: int | None,
    ) -> None:
        elapsed = max(time.monotonic() - self.started_monotonic, 0.0)
        metric = TrainingMetric(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            global_timestep=self.current_timestep,
            completed_episode_count=self.completed_episode_count,
            episode_reward=episode_reward,
            episode_length=episode_length,
            recent_mean_episode_reward=(
                sum(self.recent_rewards) / len(self.recent_rewards)
                if self.recent_rewards
                else None
            ),
            elapsed_seconds=elapsed,
            timesteps_per_second=(
                self.current_timestep / elapsed if elapsed > 0 else None
            ),
            training_status=status,
            timestamp=datetime.now(timezone.utc),
        )
        with SessionLocal() as db:
            db.add(metric)
            db.commit()
            db.refresh(metric)
            payload = TrainingMetricResponse.model_validate(metric).model_dump(
                mode="json"
            )
        metric_stream.publish(
            self.experiment_id,
            {"type": "metric", "metric": payload},
        )

    def finalize(self, status: str) -> None:
        if getattr(self, "model", None) is not None:
            self._persist_metric(status, episode_reward=None, episode_length=None)
