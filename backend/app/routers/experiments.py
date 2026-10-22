import asyncio
import logging
from queue import Empty
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import Experiment, PolicyEvaluation, TrainingMetric, TrainingRun
from app.schemas import ExperimentActionResponse, ExperimentCreate, ExperimentResponse
from app.schemas.metrics import TrainingMetricResponse
from app.schemas.runs import (
    EvaluationRequest,
    PolicyEvaluationResponse,
)
from app.services.evaluation_service import launch_evaluation
from app.services.metric_stream import metric_stream, publish_status
from app.services.training_service import (
    MAX_RUNS_PER_EXPERIMENT,
    MODEL_DIR,
    launch_job,
    release_job,
    request_cancellation,
    reserve_job,
)


router = APIRouter(prefix="/experiments", tags=["Experiments"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(
    experiment_data: ExperimentCreate,
    db: Session = Depends(get_db),
):
    experiment = Experiment(
        name=experiment_data.name,
        environment=experiment_data.environment,
        algorithm=experiment_data.algorithm,
        total_timesteps=experiment_data.total_timesteps,
        learning_rate=experiment_data.learning_rate,
        n_steps=experiment_data.n_steps,
        batch_size=experiment_data.batch_size,
        gamma=experiment_data.gamma,
        n_epochs=experiment_data.n_epochs,
        base_seed=experiment_data.base_seed,
        run_count=experiment_data.run_count,
        status="created",
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    return db.query(Experiment).order_by(Experiment.created_at.desc()).all()


@router.get("/{experiment_id}/metrics", response_model=list[TrainingMetricResponse])
def get_experiment_metrics(
    experiment_id: int = Path(ge=1),
    limit: int = Query(default=1000, ge=1, le=10000),
    from_timestep: int | None = Query(default=None, ge=0),
    to_timestep: int | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
):
    if from_timestep is not None and to_timestep is not None:
        if from_timestep > to_timestep:
            raise HTTPException(
                status_code=422,
                detail="from_timestep must be less than or equal to to_timestep",
            )
    if db.get(Experiment, experiment_id) is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    query = db.query(TrainingMetric).filter(
        TrainingMetric.experiment_id == experiment_id
    )
    if from_timestep is not None:
        query = query.filter(TrainingMetric.global_timestep >= from_timestep)
    if to_timestep is not None:
        query = query.filter(TrainingMetric.global_timestep <= to_timestep)
    recent_metrics = (
        query.order_by(
            TrainingMetric.timestamp.desc(),
            TrainingMetric.id.desc(),
        )
        .limit(limit)
        .all()
    )
    return sorted(
        recent_metrics,
        key=lambda metric: (
            metric.global_timestep,
            metric.timestamp,
            metric.id,
        ),
    )


@router.get(
    "/{experiment_id}/runs/{run_id}/metrics",
    response_model=list[TrainingMetricResponse],
)
def get_run_metrics(
    experiment_id: int = Path(ge=1),
    run_id: int = Path(ge=1),
    limit: int = Query(default=1000, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    run = db.get(TrainingRun, run_id)
    if run is None or run.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail="Training run not found")
    recent_metrics = (
        db.query(TrainingMetric)
        .filter(TrainingMetric.run_id == run_id)
        .order_by(
            TrainingMetric.global_timestep.desc(),
            TrainingMetric.timestamp.desc(),
            TrainingMetric.id.desc(),
        )
        .limit(limit)
        .all()
    )
    return sorted(
        recent_metrics,
        key=lambda metric: (
            metric.global_timestep,
            metric.timestamp,
            metric.id,
        ),
    )


@router.post(
    "/runs/{run_id}/evaluate",
    response_model=PolicyEvaluationResponse,
    status_code=202,
)
def start_run_evaluation(
    run_id: int = Path(ge=1),
    request: EvaluationRequest = EvaluationRequest(),
    db: Session = Depends(get_db),
):
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Only successfully completed training runs can be evaluated",
        )
    existing_evaluation = (
        db.query(PolicyEvaluation).filter_by(run_id=run_id).one_or_none()
    )
    if existing_evaluation is not None:
        raise HTTPException(
            status_code=409,
            detail="This training run already has a policy evaluation",
        )
    artifact = MODEL_DIR / f"run_{run_id}.zip"
    if not artifact.is_file():
        raise HTTPException(
            status_code=409,
            detail="The saved policy artifact is unavailable for this run",
        )

    evaluation = PolicyEvaluation(
        run_id=run_id,
        status="pending",
        episode_count=request.episode_count,
        base_seed=request.base_seed,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    try:
        launch_evaluation(evaluation.id)
    except Exception as error:
        logger.exception(
            "Could not queue policy evaluation for run_id=%s evaluation_id=%s",
            run_id,
            evaluation.id,
        )
        evaluation.status = "failed"
        evaluation.failure_message = "The evaluation worker could not be started."
        evaluation.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(
            status_code=503,
            detail="Evaluation could not be queued. Check server logs and retry.",
        ) from error
    return evaluation


@router.get(
    "/runs/{run_id}/evaluation",
    response_model=PolicyEvaluationResponse,
)
def get_run_evaluation(run_id: int = Path(ge=1), db: Session = Depends(get_db)):
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.evaluation is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return run.evaluation


@router.websocket("/{experiment_id}/metrics/ws")
async def experiment_metrics_websocket(
    websocket: WebSocket,
    experiment_id: int,
    db: Session = Depends(get_db),
):
    if experiment_id < 1:
        await websocket.close(code=4400)
        return
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        await websocket.close(code=4404)
        return

    try:
        subscriber = metric_stream.subscribe(experiment_id)
    except RuntimeError:
        db.close()
        await websocket.close(code=1013)
        return

    event_task = None
    receive_task = None
    try:
        db.expire_all()
        experiment = db.get(Experiment, experiment_id)
        if experiment is None:
            await websocket.close(code=4404)
            return
        status = experiment.status
        total_timesteps = experiment.total_timesteps
        latest_metric = (
            db.query(TrainingMetric)
            .filter(TrainingMetric.experiment_id == experiment_id)
            .order_by(
                TrainingMetric.timestamp.desc(),
                TrainingMetric.id.desc(),
            )
            .first()
        )
        latest_payload = (
            TrainingMetricResponse.model_validate(latest_metric).model_dump(
                mode="json"
            )
            if latest_metric is not None
            else None
        )
        db.close()
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "status",
                "experiment_id": experiment_id,
                "status": status,
                "total_timesteps": total_timesteps,
                "latest_metric": latest_payload,
                "history_url": f"/experiments/{experiment_id}/metrics",
            }
        )
        if status in {"completed", "failed", "stopped"}:
            await websocket.close(code=1000)
            return

        async def wait_for_event():
            while True:
                try:
                    return subscriber.get_nowait()
                except Empty:
                    await asyncio.sleep(0.25)

        event_task = asyncio.create_task(wait_for_event())
        receive_task = asyncio.create_task(websocket.receive_text())
        while True:
            completed, _ = await asyncio.wait(
                {event_task, receive_task},
                timeout=15,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not completed:
                await websocket.send_json({"type": "heartbeat"})
                continue
            if receive_task in completed:
                receive_task.result()
                receive_task = asyncio.create_task(websocket.receive_text())
            if event_task in completed:
                await websocket.send_json(event_task.result())
                event_task = asyncio.create_task(wait_for_event())
    except WebSocketDisconnect:
        pass
    finally:
        pending_tasks = [
            task
            for task in (event_task, receive_task)
            if task is not None and not task.done()
        ]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        metric_stream.unsubscribe(experiment_id, subscriber)
        db.close()


@router.post("/{experiment_id}/start", response_model=ExperimentActionResponse)
def start_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
):
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if any(
        value is None
        for value in (
            experiment.learning_rate,
            experiment.n_steps,
            experiment.batch_size,
            experiment.gamma,
            experiment.n_epochs,
        )
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This legacy experiment has no saved PPO configuration and cannot "
                "be restarted. Create a new experiment to configure its training."
            ),
        )
    if experiment.base_seed is None or experiment.run_count is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This legacy experiment has no recorded seed configuration. "
                "Create a new experiment to configure reproducible runs."
            ),
        )

    try:
        cancel_event = reserve_job(experiment_id)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    try:
        existing_run_count = (
            db.query(TrainingRun).filter_by(experiment_id=experiment_id).count()
        )
        if existing_run_count + experiment.run_count > MAX_RUNS_PER_EXPERIMENT:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"An experiment can contain at most "
                    f"{MAX_RUNS_PER_EXPERIMENT} training runs."
                ),
            )
        result = db.execute(
            update(Experiment)
            .where(
                Experiment.id == experiment_id,
                Experiment.status.in_(["created", "failed", "stopped"]),
                Experiment.algorithm == "PPO",
                Experiment.environment == "InvertedPendulum-v5",
            )
            .values(
                status="pending",
                completed_at=None,
                failure_message=None,
            )
        )
        if result.rowcount != 1:
            db.rollback()
            current = db.get(Experiment, experiment_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Experiment not found")
            if current.algorithm != "PPO" or current.environment != "InvertedPendulum-v5":
                raise HTTPException(
                    status_code=422,
                    detail="Only PPO with InvertedPendulum-v5 can be started",
                )
            raise HTTPException(
                status_code=409,
                detail=f"Cannot start an experiment in state '{current.status}'",
            )

        next_run_index = existing_run_count
        for run_index in range(next_run_index, next_run_index + experiment.run_count):
            db.add(
                TrainingRun(
                    experiment_id=experiment_id,
                    run_index=run_index,
                    effective_seed=(experiment.base_seed + run_index) % (2**32),
                    status="pending",
                    requested_timesteps=experiment.total_timesteps,
                )
            )
        db.commit()
        experiment = db.get(Experiment, experiment_id)
        publish_status(experiment_id, "pending", experiment.total_timesteps)
        launch_job(experiment_id, cancel_event)
        response_data = ExperimentResponse.model_validate(experiment).model_dump()
        return ExperimentActionResponse(
            **response_data,
            message="Training accepted and queued.",
        )
    except HTTPException:
        db.rollback()
        release_job(experiment_id)
        raise
    except Exception as error:
        logger.exception(
            "Could not queue experiment training for experiment_id=%s",
            experiment_id,
        )
        db.rollback()
        release_job(experiment_id)
        with SessionLocal() as status_db:
            try:
                failure_message = (
                    f"Training worker could not start ({type(error).__name__})."
                )
                now = datetime.now(timezone.utc)
                status_db.query(TrainingRun).filter(
                    TrainingRun.experiment_id == experiment_id,
                    TrainingRun.status == "pending",
                ).update(
                    {
                        TrainingRun.status: "failed",
                        TrainingRun.completed_at: now,
                        TrainingRun.failure_message: failure_message,
                    },
                    synchronize_session=False,
                )
                status_db.query(Experiment).filter(
                    Experiment.id == experiment_id,
                    Experiment.status == "pending",
                ).update(
                    {
                        Experiment.status: "failed",
                        Experiment.completed_at: now,
                        Experiment.failure_message: failure_message,
                    },
                    synchronize_session=False,
                )
                status_db.commit()
            except Exception:
                status_db.rollback()
                logger.exception(
                    "Could not persist training launch failure for experiment_id=%s",
                    experiment_id,
                )
                raise HTTPException(
                    status_code=503,
                    detail="Training could not be queued and its status could not be updated.",
                ) from error
        publish_status(experiment_id, "failed", experiment.total_timesteps)
        raise HTTPException(
            status_code=503,
            detail="Training could not be queued. Check server logs and retry.",
        ) from error


@router.post("/{experiment_id}/stop", response_model=ExperimentActionResponse)
def stop_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
):
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.status not in {"pending", "running"}:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot stop an experiment in state '{experiment.status}'",
        )
    if not request_cancellation(experiment_id):
        raise HTTPException(
            status_code=409,
            detail="No active training job is available to acknowledge cancellation",
        )

    result = db.execute(
        update(Experiment)
        .where(
            Experiment.id == experiment_id,
            Experiment.status.in_(["pending", "running"]),
        )
        .values(status="stopping")
    )
    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=409, detail="Experiment state changed; refresh and retry")

    db.commit()
    experiment = db.get(Experiment, experiment_id)
    publish_status(experiment_id, "stopping", experiment.total_timesteps)
    response_data = ExperimentResponse.model_validate(experiment).model_dump()
    return ExperimentActionResponse(
        **response_data,
        message="Cancellation requested; waiting for PPO to stop.",
    )
