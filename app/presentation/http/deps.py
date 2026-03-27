from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.services.factory import create_ingestion_service, create_rag_service
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db_session():
        yield session


async def get_ingestion_service(
    session: AsyncSession = Depends(get_session),
) -> IngestionService:
    return create_ingestion_service(session)


async def get_rag_service(
    session: AsyncSession = Depends(get_session),
) -> RAGService:
    return create_rag_service(session)

