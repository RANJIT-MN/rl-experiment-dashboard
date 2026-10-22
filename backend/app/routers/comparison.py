import csv
import io
import statistics
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import EvaluationEpisode, Experiment, TrainingMetric, TrainingRun
from app.schemas.comparison import (
    ExperimentComparisonItem,
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
)
from app.schemas.experiment import ExperimentResponse
from app.schemas.metrics import TrainingMetricResponse


router = APIRouter(prefix="/experiments", tags=["Experiment comparison"])
MAX_CURVE_POINTS = 2000
CSV_BATCH_SIZE = 500
METRIC_DEFINITION = (
    "Episode reward and length summaries use every persisted completed-episode "
    "sample across the selected experiment's history; training speed is the "
    "most recent persisted non-null measurement. Separate evaluation reward "
    "means summarize each completed run's policy evaluation, with sample "
    "standard deviation across run means only when at least two runs were "
    "evaluated; evaluation episode length is pooled over completed evaluations."
)
CSV_HEADERS = [
    "experiment_id",
    "experiment_name",
    "timestep",
    "completed_episode_count",
    "episode_reward",
    "episode_length",
    "recent_mean_episode_reward",
    "training_speed",
    "training_status",
    "timestamp",
    "learning_rate",
    "n_steps",
    "batch_size",
    "gamma",
    "n_epochs",
    "run_id",
    "run_index",
    "effective_seed",
]


def _get_selected_experiments(
    experiment_ids: list[int],
    db: Session,
) -> list[Experiment]:
    experiments = (
        db.query(Experiment)
        .filter(Experiment.id.in_(experiment_ids))
        .all()
    )
    by_id = {experiment.id: experiment for experiment in experiments}
    missing = [experiment_id for experiment_id in experiment_ids if experiment_id not in by_id]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment(s) not found: {', '.join(map(str, missing))}",
        )
    return [by_id[experiment_id] for experiment_id in experiment_ids]


@router.post("/compare", response_model=ExperimentComparisonResponse)
def compare_experiments(
    request: ExperimentComparisonRequest,
    db: Session = Depends(get_db),
):
    experiments = _get_selected_experiments(request.experiment_ids, db)
    results = []

    for experiment in experiments:
        rewards = (
            db.query(
                func.count(TrainingMetric.id),
                func.avg(TrainingMetric.episode_reward),
                func.max(TrainingMetric.episode_reward),
                func.min(TrainingMetric.episode_reward),
            )
            .filter(
                TrainingMetric.experiment_id == experiment.id,
                TrainingMetric.episode_reward.is_not(None),
            )
            .one()
        )
        lengths = (
            db.query(func.avg(TrainingMetric.episode_length))
            .filter(
                TrainingMetric.experiment_id == experiment.id,
                TrainingMetric.episode_length.is_not(None),
            )
            .scalar()
        )
        latest_reward = (
            db.query(TrainingMetric.episode_reward)
            .filter(
                TrainingMetric.experiment_id == experiment.id,
                TrainingMetric.episode_reward.is_not(None),
            )
            .order_by(
                TrainingMetric.timestamp.desc(),
                TrainingMetric.id.desc(),
            )
            .limit(1)
            .scalar()
        )
        latest_speed = (
            db.query(TrainingMetric.timesteps_per_second)
            .filter(
                TrainingMetric.experiment_id == experiment.id,
                TrainingMetric.timesteps_per_second.is_not(None),
            )
            .order_by(
                TrainingMetric.timestamp.desc(),
                TrainingMetric.id.desc(),
            )
            .limit(1)
            .scalar()
        )
        recent_curve_metrics = (
            db.query(TrainingMetric)
            .filter(
                TrainingMetric.experiment_id == experiment.id,
                or_(
                    TrainingMetric.episode_reward.is_not(None),
                    TrainingMetric.episode_length.is_not(None),
                ),
            )
            .order_by(
                TrainingMetric.timestamp.desc(),
                TrainingMetric.id.desc(),
            )
            .limit(MAX_CURVE_POINTS + 1)
            .all()
        )
        curve_truncated = len(recent_curve_metrics) > MAX_CURVE_POINTS
        curve_metrics = sorted(
            recent_curve_metrics[:MAX_CURVE_POINTS],
            key=lambda metric: (
                metric.global_timestep,
                metric.timestamp,
                metric.id,
            ),
        )
        runs = (
            db.query(TrainingRun)
            .filter(TrainingRun.experiment_id == experiment.id)
            .order_by(TrainingRun.run_index)
            .all()
        )
        completed_evaluations = [
            run.evaluation
            for run in runs
            if run.evaluation is not None and run.evaluation.status == "completed"
        ]
        run_evaluation_rewards = [
            evaluation.mean_episode_reward
            for evaluation in completed_evaluations
            if evaluation.mean_episode_reward is not None
        ]
        evaluation_ids = [evaluation.id for evaluation in completed_evaluations]
        evaluation_lengths = (
            [
                row.length
                for row in db.query(EvaluationEpisode.length)
                .filter(EvaluationEpisode.evaluation_id.in_(evaluation_ids))
                .all()
            ]
            if evaluation_ids
            else []
        )
        results.append(
            ExperimentComparisonItem(
                experiment=ExperimentResponse.model_validate(experiment),
                completed_episode_count=rewards[0],
                latest_episode_reward=latest_reward,
                mean_episode_reward=rewards[1],
                maximum_episode_reward=rewards[2],
                minimum_episode_reward=rewards[3],
                mean_episode_length=lengths,
                latest_training_speed=latest_speed,
                curve_metrics=[
                    TrainingMetricResponse.model_validate(metric)
                    for metric in curve_metrics
                ],
                curve_truncated=curve_truncated,
                training_run_count=len(runs),
                completed_training_run_count=sum(
                    run.status == "completed" for run in runs
                ),
                evaluated_run_count=len(run_evaluation_rewards),
                evaluation_reward_mean=(
                    statistics.mean(run_evaluation_rewards)
                    if run_evaluation_rewards
                    else None
                ),
                evaluation_reward_sample_stddev=(
                    statistics.stdev(run_evaluation_rewards)
                    if len(run_evaluation_rewards) > 1
                    else None
                ),
                evaluation_episode_count=len(evaluation_lengths),
                mean_evaluation_episode_length=(
                    statistics.mean(evaluation_lengths)
                    if evaluation_lengths
                    else None
                ),
            )
        )

    return ExperimentComparisonResponse(
        experiments=results,
        metric_definition=METRIC_DEFINITION,
    )


def _csv_value(value):
    return "" if value is None else value


def _write_csv_row(values: list) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\r\n").writerow(
        [_csv_value(value) for value in values]
    )
    return buffer.getvalue().encode("utf-8")


def _stream_metric_csv(experiment_ids: list[int]):
    db = SessionLocal()
    try:
        experiments = _get_selected_experiments(experiment_ids, db)
        experiment_by_id = {experiment.id: experiment for experiment in experiments}
        run_by_id = {
            run.id: run
            for run in db.query(TrainingRun)
            .filter(TrainingRun.experiment_id.in_(experiment_ids))
            .all()
        }
        yield b"\xef\xbb\xbf" + _write_csv_row(CSV_HEADERS)
        metrics = (
            db.query(TrainingMetric)
            .filter(TrainingMetric.experiment_id.in_(experiment_ids))
            .order_by(
                TrainingMetric.experiment_id.asc(),
                TrainingMetric.global_timestep.asc(),
                TrainingMetric.timestamp.asc(),
                TrainingMetric.id.asc(),
            )
            .yield_per(CSV_BATCH_SIZE)
        )
        for metric in metrics:
            experiment = experiment_by_id[metric.experiment_id]
            run = run_by_id.get(metric.run_id)
            yield _write_csv_row(
                [
                    metric.experiment_id,
                    experiment.name,
                    metric.global_timestep,
                    metric.completed_episode_count,
                    metric.episode_reward,
                    metric.episode_length,
                    metric.recent_mean_episode_reward,
                    metric.timesteps_per_second,
                    metric.training_status,
                    (
                        metric.timestamp.replace(tzinfo=timezone.utc)
                        if metric.timestamp.tzinfo is None
                        else metric.timestamp
                    ).isoformat(),
                    experiment.learning_rate,
                    experiment.n_steps,
                    experiment.batch_size,
                    experiment.gamma,
                    experiment.n_epochs,
                    metric.run_id,
                    run.run_index if run is not None else None,
                    run.effective_seed if run is not None else None,
                ]
            )
    finally:
        db.close()


@router.post("/compare/export")
def export_comparison_metrics(
    request: ExperimentComparisonRequest,
    db: Session = Depends(get_db),
):
    _get_selected_experiments(request.experiment_ids, db)
    filename = datetime.now(timezone.utc).strftime(
        "experiment-comparison-%Y%m%d-%H%M%S.csv"
    )
    return StreamingResponse(
        _stream_metric_csv(request.experiment_ids),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
