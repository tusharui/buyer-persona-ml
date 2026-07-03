import asyncio
import traceback

from src.celery_app import celery_app
from src.pipeline import async_main


@celery_app.task(name="train_pipeline", autoretry_for=(Exception,), max_retries=2)
def train_pipeline_task(source: str = "neon"):
    try:
        result = asyncio.run(async_main(source=source))
        return {
            "status": "completed",
            "customers": len(result),
            "clusters": int(result["Cluster"].nunique()),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
