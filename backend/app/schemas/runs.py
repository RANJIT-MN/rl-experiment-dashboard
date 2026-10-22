from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PolicyEvaluationSummaryResponse(BaseModel):
    id: int
    status: str
    episode_count: int
    base_seed: int
    mean_episode_reward: float | None = None
    mean_episode_length: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_message: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ExperimentRunResponse(BaseModel):
    id: int
    experiment_id: int
    run_index: int
    effective_seed: int
    status: str
    requested_timesteps: int
    actual_timesteps: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_message: str | None = None
    evaluation: PolicyEvaluationSummaryResponse | None = None

    model_config = ConfigDict(from_attributes=True)


class EvaluationEpisodeResponse(BaseModel):
    episode_index: int
    seed: int
    reward: float
    length: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class PolicyEvaluationResponse(PolicyEvaluationSummaryResponse):
    episodes: list[EvaluationEpisodeResponse] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    episode_count: int = Field(default=10, ge=1, le=100, strict=True)
    base_seed: int = Field(default=100_000, ge=0, le=4_294_967_295, strict=True)
