from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    name = Column(
        String,
        nullable=False,
    )

    environment = Column(
        String,
        nullable=False,
        default="InvertedPendulum-v5",
    )

    algorithm = Column(
        String,
        nullable=False,
        default="PPO",
    )

    status = Column(
        String,
        nullable=False,
        default="created",
    )

    failure_message = Column(
        Text,
        nullable=True,
    )

    total_timesteps = Column(
        Integer,
        nullable=False,
        default=0,
    )

    learning_rate = Column(Float, nullable=True, default=3e-4)
    n_steps = Column(Integer, nullable=True, default=2048)
    batch_size = Column(Integer, nullable=True, default=64)
    gamma = Column(Float, nullable=True, default=0.99)
    n_epochs = Column(Integer, nullable=True, default=10)
    base_seed = Column(Integer, nullable=True, default=42)
    run_count = Column(Integer, nullable=True, default=3)

    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    completed_at = Column(
        DateTime,
        nullable=True,
    )

    metrics = relationship(
        "TrainingMetric",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )
    runs = relationship(
        "TrainingRun",
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="TrainingRun.run_index",
        lazy="selectin",
    )