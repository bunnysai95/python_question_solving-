from celery import Celery
import os
from settings import settings

# Broker URL: prefer explicit env var REDIS_URL or settings.CELERY_BROKER_URL, fallback to redis service name
broker = os.getenv("REDIS_URL") or getattr(settings, "CELERY_BROKER_URL", "redis://redis:6379/0")

celery_app = Celery(
    "fastapi_app",
    broker=broker,
)

# simple config: use JSON serializer for better interoperability
celery_app.conf.update(
    result_serializer="json",
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
)

