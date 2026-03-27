from __future__ import annotations

import asyncio
from uuid import UUID

from app.db.session import AsyncSessionLocal
from app.services.factory import create_ingestion_service
from app.tasks.celery_app import celery_app


@celery_app.task(name="app.tasks.document_tasks.process_document_task", bind=True)
def process_document_task(self, document_id: str) -> str:  # noqa: ARG001
    async def _run() -> None:
        async with AsyncSessionLocal() as session:
            ingestion_service = create_ingestion_service(session)
            await ingestion_service.ingest_document(UUID(document_id))

    asyncio.run(_run())
    return document_id
