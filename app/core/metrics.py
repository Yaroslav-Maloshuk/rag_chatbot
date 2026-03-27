from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings


def setup_metrics(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.enable_metrics:
        return
    Instrumentator(excluded_handlers=["/health/live"]).instrument(app).expose(app)
