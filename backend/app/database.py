import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "rl_dashboard.db"
DATABASE_URL = os.getenv(
    "RL_DASHBOARD_DATABASE_URL",
    f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}",
)


# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


# Create database session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# Base class for SQLAlchemy models
Base = declarative_base()


def initialize_database():
    Base.metadata.create_all(bind=engine)
    columns = {
        column["name"]
        for column in inspect(engine).get_columns("experiments")
    }
    additive_columns = {
        "failure_message": "TEXT",
        "learning_rate": "FLOAT",
        "n_steps": "INTEGER",
        "batch_size": "INTEGER",
        "gamma": "FLOAT",
        "n_epochs": "INTEGER",
        "base_seed": "INTEGER",
        "run_count": "INTEGER",
    }
    missing_columns = {
        name: column_type
        for name, column_type in additive_columns.items()
        if name not in columns
    }
    if missing_columns:
        with engine.begin() as connection:
            for name, column_type in missing_columns.items():
                connection.execute(
                    text(
                        f"ALTER TABLE experiments "
                        f"ADD COLUMN {name} {column_type}"
                    )
                )

    metric_columns = {
        column["name"]
        for column in inspect(engine).get_columns("training_metrics")
    }
    if "run_id" not in metric_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE training_metrics "
                    "ADD COLUMN run_id INTEGER REFERENCES training_runs(id) "
                    "ON DELETE CASCADE"
                )
            )
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS "
                "ix_training_metrics_run_timestep_timestamp "
                "ON training_metrics "
                "(run_id, global_timestep, timestamp, id)"
            )
        )


# Dependency for database sessions
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()