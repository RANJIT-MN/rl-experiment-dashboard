import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import gymnasium as gym
from stable_baselines3 import PPO

from app.database import SessionLocal
from app.models import EvaluationEpisode, Experiment, PolicyEvaluation, TrainingRun
from app.services.compute_slot import compute_slot
from app.services.training_service import MODEL_DIR


_EVALUATION_JOBS: dict[int, tuple[threading.Thread, threading.Event]] = {}
_EVALUATION_LOCK = threading.Lock()
MAX_QUEUED_EVALUATIONS = 10
logger = logging.getLogger(__name__)


def launch_evaluation(evaluation_id: int) -> None:
    cancel_event = threading.Event()

    def worker():
        try:
            evaluate_policy(evaluation_id, cancel_event)
        finally:
            with _EVALUATION_LOCK:
                _EVALUATION_JOBS.pop(evaluation_id, None)

    thread = threading.Thread(
        target=worker,
        name=f"evaluation-{evaluation_id}",
        daemon=True,
    )
    with _EVALUATION_LOCK:
        if evaluation_id in _EVALUATION_JOBS:
            raise RuntimeError("Evaluation is already scheduled")
        if len(_EVALUATION_JOBS) >= MAX_QUEUED_EVALUATIONS:
            raise RuntimeError("The bounded evaluation queue is full; try again later")
        _EVALUATION_JOBS[evaluation_id] = (thread, cancel_event)
    try:
        thread.start()
    except Exception:
        with _EVALUATION_LOCK:
            _EVALUATION_JOBS.pop(evaluation_id, None)
        raise


def evaluate_policy(
    evaluation_id: int,
    cancel_event: threading.Event | None = None,
) -> None:
    cancel_event = cancel_event or threading.Event()
    while not cancel_event.is_set():
        if compute_slot.acquire(timeout=0.1):
            break
    else:
        with SessionLocal() as db:
            evaluation = db.get(PolicyEvaluation, evaluation_id)
            if evaluation is not None and evaluation.status == "pending":
                evaluation.status = "stopped"
                evaluation.failure_message = (
                    "Evaluation stopped before it acquired the compute slot."
                )
                evaluation.completed_at = datetime.now(timezone.utc)
                db.commit()
        return
    try:
        _evaluate_policy_with_slot(evaluation_id, cancel_event)
    finally:
        compute_slot.release()


def _evaluate_policy_with_slot(
    evaluation_id: int,
    cancel_event: threading.Event,
) -> None:
    env = None
    with SessionLocal() as db:
        evaluation = db.get(PolicyEvaluation, evaluation_id)
        if evaluation is None:
            return
        run = db.get(TrainingRun, evaluation.run_id)
        experiment = db.get(Experiment, run.experiment_id) if run else None
        if run is None or experiment is None:
            evaluation.status = "failed"
            evaluation.failure_message = "The training run no longer exists."
            evaluation.completed_at = datetime.now(timezone.utc)
            db.commit()
            return
        evaluation.status = "running"
        evaluation.started_at = datetime.now(timezone.utc)
        run_id = run.id
        environment_name = experiment.environment
        episode_count = evaluation.episode_count
        base_seed = evaluation.base_seed
        db.commit()

    rewards = []
    lengths = []
    status = "failed"
    failure_message = None
    try:
        artifact = MODEL_DIR / f"run_{run_id}.zip"
        if not artifact.is_file():
            raise FileNotFoundError("The saved policy artifact is unavailable.")
        env = gym.make(environment_name)
        model = PPO.load(str(artifact), env=env, device="cpu")
        for episode_index in range(episode_count):
            if cancel_event.is_set():
                status = "stopped"
                failure_message = "Evaluation stopped during backend shutdown."
                break
            seed = (base_seed + episode_index) % (2**32)
            observation, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            reward_sum = 0.0
            length = 0
            while not (terminated or truncated):
                if cancel_event.is_set():
                    status = "stopped"
                    failure_message = "Evaluation stopped during backend shutdown."
                    break
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                length += 1
            if cancel_event.is_set():
                break
            with SessionLocal() as db:
                db.add(
                    EvaluationEpisode(
                        evaluation_id=evaluation_id,
                        episode_index=episode_index,
                        seed=seed,
                        reward=reward_sum,
                        length=length,
                    )
                )
                db.commit()
            rewards.append(reward_sum)
            lengths.append(length)
        if status != "stopped":
            status = "completed"
    except Exception as error:
        failure_message = f"Evaluation failed ({type(error).__name__})."
        logger.exception("Policy evaluation failed for evaluation_id=%s", evaluation_id)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                if status == "completed":
                    status = "failed"
                    failure_message = (
                        f"Evaluation environment cleanup failed ({type(error).__name__})."
                    )
                logger.exception(
                    "Evaluation environment cleanup failed for evaluation_id=%s",
                    evaluation_id,
                )
        with SessionLocal() as db:
            evaluation = db.get(PolicyEvaluation, evaluation_id)
            if evaluation is not None:
                evaluation.status = status
                evaluation.mean_episode_reward = (
                    mean(rewards) if rewards and status == "completed" else None
                )
                evaluation.mean_episode_length = (
                    mean(lengths) if lengths and status == "completed" else None
                )
                evaluation.failure_message = failure_message
                evaluation.completed_at = datetime.now(timezone.utc)
                db.commit()


def shutdown_evaluations(timeout: float = 5.0) -> None:
    with _EVALUATION_LOCK:
        jobs = list(_EVALUATION_JOBS.values())
        for _, cancel_event in jobs:
            cancel_event.set()
    deadline = datetime.now(timezone.utc).timestamp() + timeout
    for thread, _ in jobs:
        if thread is threading.current_thread():
            continue
        remaining = max(0.0, deadline - datetime.now(timezone.utc).timestamp())
        thread.join(remaining)
