import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from sqlalchemy import update

from app.database import SessionLocal
from app.models import Experiment, TrainingRun
from app.services.compute_slot import compute_slot
from app.services.metric_stream import publish_status
from app.services.training_metrics import TrainingMetricsCallback


logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[2] / "trained_models"
_ALLOWED_ENVIRONMENT = "InvertedPendulum-v5"
_JOBS: dict[int, tuple[threading.Event, threading.Thread | None, bool]] = {}
_JOBS_LOCK = threading.Lock()
_SHUTTING_DOWN = False
MAX_RUNS_PER_EXPERIMENT = 100
MAX_QUEUED_EXPERIMENTS = 10


class TrainingCancelled(Exception):
    """Raised after PPO acknowledges a cooperative cancellation request."""


def reserve_job(experiment_id: int) -> threading.Event:
    global _SHUTTING_DOWN
    with _JOBS_LOCK:
        if _SHUTTING_DOWN:
            raise RuntimeError("The training service is shutting down")
        if experiment_id in _JOBS:
            raise RuntimeError("Training is already scheduled for this experiment")
        if len(_JOBS) >= MAX_QUEUED_EXPERIMENTS:
            raise RuntimeError("The bounded training queue is full; try again later")
        cancel_event = threading.Event()
        _JOBS[experiment_id] = (cancel_event, None, True)
        return cancel_event


def attach_job_thread(experiment_id: int, thread: threading.Thread) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(experiment_id)
        if job is not None:
            _JOBS[experiment_id] = (job[0], thread, job[2])


def launch_job(experiment_id: int, cancel_event: threading.Event) -> None:
    thread = threading.Thread(
        target=run_experiment,
        args=(experiment_id, cancel_event),
        name=f"experiment-{experiment_id}",
        daemon=True,
    )
    attach_job_thread(experiment_id, thread)
    try:
        thread.start()
    except Exception:
        release_job(experiment_id)
        raise


def release_job(experiment_id: int) -> None:
    with _JOBS_LOCK:
        _JOBS.pop(experiment_id, None)


def request_cancellation(experiment_id: int) -> bool:
    with _JOBS_LOCK:
        job = _JOBS.get(experiment_id)
        if job is None or not job[2]:
            return False
        job[0].set()
        return True


def close_cancellation_window(experiment_id: int) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(experiment_id)
        if job is not None:
            _JOBS[experiment_id] = (job[0], job[1], False)


def _now():
    return datetime.now(timezone.utc)


def _set_run_status(
    run_id: int,
    status: str,
    *,
    actual_timesteps: int | None = None,
    failure_message: str | None = None,
    start: bool = False,
) -> None:
    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        allowed_previous = {
            "running": {"pending"},
            "completed": {"running"},
            "failed": {"pending", "running"},
            "stopped": {"pending", "running"},
        }.get(status, set())
        if run.status == status:
            return
        if run.status not in allowed_previous:
            logger.warning(
                "Ignoring invalid training run transition run_id=%s %s->%s",
                run_id,
                run.status,
                status,
            )
            return
        run.status = status
        if start:
            run.started_at = _now()
        if status in {"completed", "failed", "stopped"}:
            run.completed_at = _now()
        if actual_timesteps is not None:
            run.actual_timesteps = actual_timesteps
        run.failure_message = failure_message
        db.commit()


def _set_experiment_status(
    experiment_id: int,
    status: str,
    failure_message: str | None = None,
) -> None:
    with SessionLocal() as db:
        experiment = db.get(Experiment, experiment_id)
        if experiment is None:
            return
        allowed_previous = {
            "pending": {"created", "failed", "stopped"},
            "running": {"pending"},
            "completed": {"pending", "running"},
            "failed": {"pending", "running", "stopping"},
            "stopped": {"pending", "running", "stopping"},
        }.get(status, set())
        if experiment.status == status:
            return
        if experiment.status not in allowed_previous:
            logger.warning(
                "Ignoring invalid experiment transition experiment_id=%s %s->%s",
                experiment_id,
                experiment.status,
                status,
            )
            return
        experiment.status = status
        if status in {"completed", "failed", "stopped"}:
            experiment.completed_at = _now()
        else:
            experiment.completed_at = None
        experiment.failure_message = failure_message
        total_timesteps = experiment.total_timesteps
        db.commit()
    publish_status(experiment_id, status, total_timesteps)


def _ensure_legacy_run(experiment_id: int) -> list[int]:
    """Keep direct worker invocations compatible with pre-run test fixtures."""
    with SessionLocal() as db:
        experiment = db.get(Experiment, experiment_id)
        if experiment is None:
            return []
        runs = (
            db.query(TrainingRun)
            .filter_by(experiment_id=experiment_id, status="pending")
            .order_by(TrainingRun.run_index)
            .all()
        )
        if not runs and not db.query(TrainingRun.id).filter_by(
            experiment_id=experiment_id
        ).first():
            run = TrainingRun(
                experiment_id=experiment_id,
                run_index=0,
                effective_seed=(
                    experiment.base_seed
                    if experiment.base_seed is not None
                    else 42
                ),
                status="pending",
                requested_timesteps=experiment.total_timesteps,
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return [run.id]
        return [run.id for run in runs]


def _train_run(
    experiment_id: int,
    run_id: int,
    cancel_event: threading.Event,
    metric_final_status: str,
) -> tuple[str, str | None]:
    env = None
    callback: TrainingMetricsCallback | None = None
    status = "failed"
    failure_message = None
    actual_timesteps = None
    slot_acquired = False
    try:
        while not cancel_event.is_set():
            slot_acquired = compute_slot.acquire(timeout=0.1)
            if slot_acquired:
                break
        if not slot_acquired:
            status = "stopped"
            return status, None

        with SessionLocal() as db:
            experiment = db.get(Experiment, experiment_id)
            run = db.get(TrainingRun, run_id)
            if experiment is None or run is None:
                raise ValueError("Experiment or training run no longer exists")
            if experiment.algorithm.upper() != "PPO":
                raise ValueError("Only the PPO algorithm is supported")
            if experiment.environment != _ALLOWED_ENVIRONMENT:
                raise ValueError(f"Only {_ALLOWED_ENVIRONMENT} is supported")
            if cancel_event.is_set() or experiment.status == "stopping":
                status = "stopped"
                return status, None
            started_run = db.execute(
                update(TrainingRun)
                .where(
                    TrainingRun.id == run_id,
                    TrainingRun.status == "pending",
                )
                .values(status="running", started_at=_now())
            )
            started_experiment = db.execute(
                update(Experiment)
                .where(
                    Experiment.id == experiment_id,
                    Experiment.status.in_(["pending", "running"]),
                )
                .values(status="running", completed_at=None)
            )
            if started_run.rowcount != 1 or started_experiment.rowcount != 1:
                db.rollback()
                if cancel_event.is_set():
                    status = "stopped"
                    return status, None
                raise RuntimeError(
                    "Experiment or training run changed state before the worker started"
                )
            total_timesteps = experiment.total_timesteps
            environment_name = experiment.environment
            learning_rate = experiment.learning_rate
            n_steps = experiment.n_steps
            batch_size = experiment.batch_size
            n_epochs = experiment.n_epochs
            gamma = experiment.gamma
            seed = run.effective_seed
            db.commit()

        publish_status(experiment_id, "running", total_timesteps)
        env = gym.make(environment_name)
        if isinstance(env, gym.Env):
            env = Monitor(env)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            seed=seed,
            device="cpu",
        )
        callback = TrainingMetricsCallback(
            experiment_id,
            cancel_event,
            run_id=run_id,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        actual_timesteps = getattr(model, "num_timesteps", None)
        if metric_final_status == "completed":
            close_cancellation_window(experiment_id)
        if callback.cancelled or cancel_event.is_set():
            status = "stopped"
        else:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(str(MODEL_DIR / f"run_{run_id}"))
            status = "completed"
    except Exception as error:
        failure_message = f"Training failed ({type(error).__name__})."
        logger.exception(
            "Training failed for experiment_id=%s run_id=%s",
            experiment_id,
            run_id,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                if status == "completed":
                    status = "failed"
                    failure_message = (
                        f"Environment cleanup failed: {type(error).__name__}"[:1000]
                    )
        if callback is not None:
            if actual_timesteps is None:
                actual_timesteps = callback.current_timestep
            try:
                callback.finalize(
                    status
                    if status == "stopped"
                    or (status == "failed" and metric_final_status != "running")
                    else metric_final_status
                )
            except Exception as error:
                status = "failed"
                failure_message = (
                    f"Metric persistence failed ({type(error).__name__})."
                )
        try:
            _set_run_status(
                run_id,
                status,
                actual_timesteps=actual_timesteps,
                failure_message=failure_message,
            )
        finally:
            if slot_acquired:
                compute_slot.release()
    return status, failure_message


def run_experiment(experiment_id: int, cancel_event: threading.Event | None = None):
    cancel_event = cancel_event or threading.Event()
    try:
        run_ids = _ensure_legacy_run(experiment_id)
        if not run_ids:
            return
        for run_position, run_id in enumerate(run_ids):
            if cancel_event.is_set():
                _set_run_status(run_id, "stopped")
                continue
            is_last_run = run_position == len(run_ids) - 1
            _train_run(
                experiment_id,
                run_id,
                cancel_event,
                "completed" if is_last_run else "running",
            )

        close_cancellation_window(experiment_id)
        with SessionLocal() as db:
            runs = db.query(TrainingRun).filter_by(experiment_id=experiment_id).all()
            experiment = db.get(Experiment, experiment_id)
            if experiment is None:
                return
            statuses = {run.status for run in runs}
            if "failed" in statuses:
                final_status = "failed"
                failures = [run.failure_message for run in runs if run.failure_message]
                failure_message = failures[-1] if failures else "One or more training runs failed."
            elif cancel_event.is_set() or "stopped" in statuses:
                final_status = "stopped"
                failure_message = None
            else:
                final_status = "completed"
                failure_message = None
        _set_experiment_status(experiment_id, final_status, failure_message)
    except Exception as error:
        logger.exception("Training worker failed for experiment_id=%s", experiment_id)
        try:
            with SessionLocal() as db:
                runs = (
                    db.query(TrainingRun)
                    .filter(
                        TrainingRun.experiment_id == experiment_id,
                        TrainingRun.status.in_(["pending", "running"]),
                    )
                    .all()
                )
                for run in runs:
                    run.status = "failed"
                    run.completed_at = _now()
                    run.failure_message = (
                        f"Training worker failed ({type(error).__name__})."
                    )
                db.commit()
            _set_experiment_status(
                experiment_id,
                "failed",
                f"Training worker failed ({type(error).__name__}).",
            )
        except Exception:
            logger.exception(
                "Could not persist worker failure for experiment_id=%s",
                experiment_id,
            )
    finally:
        release_job(experiment_id)


def reconcile_interrupted_work() -> None:
    """Mark process-local jobs interrupted by a server restart without resuming them."""
    from app.models import PolicyEvaluation

    interrupted_message = (
        "Interrupted by backend restart; the outcome is unknown and training "
        "was not resumed."
    )
    with SessionLocal() as db:
        interrupted_runs = (
            db.query(TrainingRun)
            .filter(TrainingRun.status.in_(["pending", "running"]))
            .all()
        )
        affected_experiment_ids = {
            run.experiment_id for run in interrupted_runs
        }
        for run in interrupted_runs:
            run.status = "failed"
            run.completed_at = _now()
            run.failure_message = interrupted_message

        active_experiments = (
            db.query(Experiment)
            .filter(Experiment.status.in_(["pending", "running", "stopping"]))
            .all()
        )
        for experiment in active_experiments:
            runs = (
                db.query(TrainingRun)
                .filter(TrainingRun.experiment_id == experiment.id)
                .all()
            )
            statuses = {run.status for run in runs}
            if runs and statuses == {"completed"}:
                experiment.status = "completed"
                experiment.failure_message = None
            elif statuses and statuses <= {"completed", "stopped"} and "stopped" in statuses:
                experiment.status = "stopped"
                experiment.failure_message = interrupted_message
            else:
                experiment.status = "failed"
                experiment.failure_message = interrupted_message
            experiment.completed_at = _now()

        active_evaluations = (
            db.query(PolicyEvaluation)
            .filter(PolicyEvaluation.status.in_(["pending", "running"]))
            .all()
        )
        for evaluation in active_evaluations:
            evaluation.status = "failed"
            evaluation.completed_at = _now()
            evaluation.failure_message = interrupted_message
        inconsistent_completed_experiments = (
            db.query(Experiment)
            .filter(
                Experiment.id.in_(affected_experiment_ids or {-1}),
                Experiment.status == "completed",
            )
            .all()
        )
        for experiment in inconsistent_completed_experiments:
            experiment.status = "failed"
            experiment.completed_at = _now()
            experiment.failure_message = interrupted_message
        final_statuses = [
            (experiment.id, experiment.status, experiment.total_timesteps)
            for experiment in [*active_experiments, *inconsistent_completed_experiments]
        ]
        db.commit()

    for experiment_id, status, total_timesteps in final_statuses:
        publish_status(
            experiment_id,
            status,
            total_timesteps,
        )


def shutdown_training(timeout: float = 5.0):
    global _SHUTTING_DOWN
    with _JOBS_LOCK:
        _SHUTTING_DOWN = True
        jobs = list(_JOBS.items())
        cancellation_ids = []
        for experiment_id, (cancel_event, _, accepting_cancellation) in jobs:
            if accepting_cancellation:
                cancel_event.set()
                cancellation_ids.append(experiment_id)

    if cancellation_ids:
        with SessionLocal() as db:
            db.query(Experiment).filter(
                Experiment.id.in_(cancellation_ids),
                Experiment.status.in_(["pending", "running"]),
            ).update({Experiment.status: "stopping"}, synchronize_session=False)
            db.query(TrainingRun).filter(
                TrainingRun.experiment_id.in_(cancellation_ids),
                TrainingRun.status == "pending",
            ).update(
                {
                    TrainingRun.status: "stopped",
                    TrainingRun.completed_at: _now(),
                    TrainingRun.failure_message: (
                        "Training stopped during backend shutdown before this run started."
                    ),
                },
                synchronize_session=False,
            )
            db.commit()

    deadline = datetime.now(timezone.utc).timestamp() + timeout
    for _, (_, thread, _) in jobs:
        if thread is None or thread is threading.current_thread():
            continue
        remaining = max(0.0, deadline - datetime.now(timezone.utc).timestamp())
        thread.join(remaining)
