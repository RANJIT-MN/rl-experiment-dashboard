from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.runs import ExperimentRunResponse


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    environment: Literal["InvertedPendulum-v5"] = "InvertedPendulum-v5"
    algorithm: Literal["PPO"] = "PPO"
    total_timesteps: int = Field(default=1000, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False, strict=True)
    n_steps: int = Field(default=2048, gt=1, strict=True)
    batch_size: int = Field(default=64, gt=1, strict=True)
    gamma: float = Field(default=0.99, gt=0, le=1, allow_inf_nan=False, strict=True)
    n_epochs: int = Field(default=10, gt=0, strict=True)
    base_seed: int = Field(default=42, ge=0, le=4_294_967_295, strict=True)
    run_count: int = Field(default=3, ge=1, le=20, strict=True)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Experiment name must not be blank")
        return value

    @model_validator(mode="after")
    def batch_size_must_match_rollout_buffer(self) -> Self:
        if self.n_steps % self.batch_size:
            raise ValueError(
                "batch_size must divide n_steps exactly for this single-environment "
                "PPO setup to avoid a truncated minibatch"
            )
        return self


class ExperimentResponse(BaseModel):
    id: int
    name: str
    environment: str
    algorithm: str
    status: Literal["created", "pending", "running", "stopping", "completed", "failed", "stopped"]
    total_timesteps: int
    created_at: datetime
    completed_at: datetime | None = None
    failure_message: str | None = None
    learning_rate: float | None = None
    n_steps: int | None = None
    batch_size: int | None = None
    gamma: float | None = None
    n_epochs: int | None = None
    base_seed: int | None = None
    run_count: int | None = None
    runs: list[ExperimentRunResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ExperimentActionResponse(ExperimentResponse):
    message: str