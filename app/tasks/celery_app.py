from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_chatbot",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.document_tasks"],
)

celery_app.conf.update(
    imports=("app.tasks.document_tasks",),
    task_routes={"app.tasks.document_tasks.process_document_task": {"queue": "documents"}},
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
