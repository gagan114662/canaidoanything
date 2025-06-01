from celery import Celery
from app.core.config import settings
import os

# Create Celery instance
celery_app = Celery(
    "garment_creative_ai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.services.tasks.image_processing",
        "app.services.tasks.ai_models"
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Import tasks
from app.services.tasks.image_processing import process_garment_image

if __name__ == "__main__":
    celery_app.start()