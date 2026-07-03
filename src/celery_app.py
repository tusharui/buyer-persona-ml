from celery import Celery

from src.config import REDIS_URL

celery_app = Celery(
    "buyer_persona",
    broker=REDIS_URL,
    backend=REDIS_URL,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)
