from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (
        UniqueConstraint("experiment_id", "run_index", name="uq_training_runs_index"),
    )

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(
        Integer,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_index = Column(Integer, nullable=False)
    effective_seed = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="pending")
    requested_timesteps = Column(Integer, nullable=False)
    actual_timesteps = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    failure_message = Column(Text, nullable=True)

    experiment = relationship("Experiment", back_populates="runs")
    metrics = relationship("TrainingMetric", back_populates="run")
    evaluation = relationship(
        "PolicyEvaluation",
        back_populates="run",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
