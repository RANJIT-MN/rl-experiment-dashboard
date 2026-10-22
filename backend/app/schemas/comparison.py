from typing import Self

from pydantic import BaseModel, Field, StrictInt, model_validator

from app.schemas.experiment import ExperimentResponse
from app.schemas.metrics import TrainingMetricResponse


class ExperimentComparisonRequest(BaseModel):
    experiment_ids: list[StrictInt] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def experiment_ids_must_be_unique(self) -> Self:
        if len(set(self.experiment_ids)) != len(self.experiment_ids):
            raise ValueError("experiment_ids must not contain duplicates")
        if any(experiment_id < 1 for experiment_id in self.experiment_ids):
            raise ValueError("experiment_ids must be positive integers")
        return self


class ExperimentComparisonItem(BaseModel):
    experiment: ExperimentResponse
    completed_episode_count: int
    latest_episode_reward: float | None
    mean_episode_reward: float | None
    maximum_episode_reward: float | None
    minimum_episode_reward: float | None
    mean_episode_length: float | None
    latest_training_speed: float | None
    curve_metrics: list[TrainingMetricResponse]
    curve_truncated: bool
    training_run_count: int = 0
    completed_training_run_count: int = 0
    evaluated_run_count: int = 0
    evaluation_reward_mean: float | None = None
    evaluation_reward_sample_stddev: float | None = None
    evaluation_episode_count: int = 0
    mean_evaluation_episode_length: float | None = None


class ExperimentComparisonResponse(BaseModel):
    experiments: list[ExperimentComparisonItem]
    metric_definition: str
