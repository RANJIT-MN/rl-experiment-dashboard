from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class PolicyEvaluation(Base):
    __tablename__ = "policy_evaluations"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_policy_evaluations_run"),
    )

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(
        Integer,
        ForeignKey("training_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    status = Column(String, nullable=False, default="pending")
    episode_count = Column(Integer, nullable=False)
    base_seed = Column(Integer, nullable=False)
    mean_episode_reward = Column(Float, nullable=True)
    mean_episode_length = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    failure_message = Column(String, nullable=True)

    run = relationship("TrainingRun", back_populates="evaluation")
    episodes = relationship(
        "EvaluationEpisode",
        back_populates="evaluation",
        cascade="all, delete-orphan",
        order_by="EvaluationEpisode.episode_index",
    )


class EvaluationEpisode(Base):
    __tablename__ = "evaluation_episodes"
    __table_args__ = (
        UniqueConstraint(
            "evaluation_id",
            "episode_index",
            name="uq_evaluation_episodes_index",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(
        Integer,
        ForeignKey("policy_evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    episode_index = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=False)
    reward = Column(Float, nullable=False)
    length = Column(Integer, nullable=False)
    timestamp = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    evaluation = relationship("PolicyEvaluation", back_populates="episodes")
