from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import initialize_database
from app.models import Experiment, TrainingMetric
from app.routers import comparison_router, experiments_router
from app.services.training_service import (
    reconcile_interrupted_work,
    shutdown_training,
)
from app.services.evaluation_service import shutdown_evaluations


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    reconcile_interrupted_work()
    try:
        yield
    finally:
        shutdown_training()
        shutdown_evaluations()


app = FastAPI(
    title="RL Experiment Dashboard API",
    description="Backend API for the Real-Time Reinforcement Learning Experiment Tracker",
    version="1.0.0",
    lifespan=lifespan,
)


# Allow the React/Vite frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(experiments_router)
app.include_router(comparison_router)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "rl-experiment-dashboard-backend",
    }