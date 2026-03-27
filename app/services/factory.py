from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.cache_service import CacheService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_pdf_service() -> PDFService:
    return PDFService()


@lru_cache(maxsize=1)
def get_chunking_service() -> ChunkingService:
    return ChunkingService()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()


@lru_cache(maxsize=1)
def get_cache_service() -> CacheService:
    return CacheService()


def create_ingestion_service(session: AsyncSession) -> IngestionService:
    return IngestionService(
        document_repository=DocumentRepository(session),
        chunk_repository=ChunkRepository(session),
        pdf_service=get_pdf_service(),
        chunking_service=get_chunking_service(),
        embedding_service=None,
    )


def create_rag_service(session: AsyncSession) -> RAGService:
    chunk_repository = ChunkRepository(session)
    retrieval_service = RetrievalService(
        chunk_repository=chunk_repository,
        embedding_service=get_embedding_service(),
    )
    return RAGService(
        retrieval_service=retrieval_service,
        llm_service=get_llm_service(),
        cache_service=get_cache_service(),
    )
