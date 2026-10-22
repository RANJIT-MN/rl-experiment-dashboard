from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_serializer


class TrainingMetricResponse(BaseModel):
    id: int
    experiment_id: int
    run_id: int | None = None
    global_timestep: int
    completed_episode_count: int
    episode_reward: float | None
    episode_length: int | None
    recent_mean_episode_reward: float | None
    elapsed_seconds: float
    timesteps_per_second: float | None
    training_status: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
