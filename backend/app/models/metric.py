from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    __table_args__ = (
        Index(
            "ix_training_metrics_experiment_timestep_timestamp",
            "experiment_id",
            "global_timestep",
            "timestamp",
            "id",
        ),
        Index(
            "ix_training_metrics_run_timestep_timestamp",
            "run_id",
            "global_timestep",
            "timestamp",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    global_timestep: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_episode_count: Mapped[int] = mapped_column(Integer, nullable=False)
    episode_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    episode_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recent_mean_episode_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    elapsed_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    timesteps_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_status: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    experiment = relationship("Experiment", back_populates="metrics")
    run = relationship("TrainingRun", back_populates="metrics")
