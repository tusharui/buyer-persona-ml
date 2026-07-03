from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from api.schemas import TrainResponse
from src.celery_app import celery_app
from src.tasks import train_pipeline_task

router = APIRouter(tags=["training"])


@router.post("/train", response_model=TrainResponse)
async def start_training(source: str = "neon"):
    try:
        task = train_pipeline_task.delay(source=source)
        return TrainResponse(
            task_id=task.id,
            status="queued",
            message="Training job queued successfully.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to queue training task: {e}",
        )


@router.get("/train/{task_id}", response_model=TrainResponse)
async def get_training_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return TrainResponse(
        task_id=task_id,
        status=result.status,
        message=(
            str(result.result) if result.ready()
            else "Task is pending."
        ),
    )
