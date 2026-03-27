from __future__ import annotations

import asyncio
import logging

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.runtime_device import get_runtime_device

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    def __init__(self) -> None:
        runtime = get_runtime_device()
        model_device = runtime.sentence_transformers_device
        logger.info("Embedding runtime backend=%s device=%s", runtime.backend, model_device)
        try:
            self._model = SentenceTransformer(settings.embedding_model_name, device=model_device)
        except Exception:  # noqa: BLE001
            logger.warning("Could not initialize embeddings on %s, falling back to CPU", model_device)
            self._model = SentenceTransformer(settings.embedding_model_name, device="cpu")
        self._query_prefix = settings.embedding_query_prefix.strip()
        self._passage_prefix = settings.embedding_passage_prefix.strip()

        model_name_lower = settings.embedding_model_name.lower()
        is_e5_family = "e5" in model_name_lower
        looks_like_e5_prefixes = (
            self._query_prefix.lower().rstrip(":") == "query"
            and self._passage_prefix.lower().rstrip(":") == "passage"
        )
        if not is_e5_family and looks_like_e5_prefixes:
            # E5-style prefixes hurt retrieval quality on non-E5 embedding models.
            logger.warning(
                "Disabling E5 prefixes for non-E5 embedding model: %s",
                settings.embedding_model_name,
            )
            self._query_prefix = ""
            self._passage_prefix = ""
        model_dim = self._model.get_sentence_embedding_dimension()
        if model_dim != settings.embedding_dim:
            logger.warning(
                "Configured embedding_dim=%s but model emits %s; update EMBEDDING_DIM",
                settings.embedding_dim,
                model_dim,
            )

    @staticmethod
    def _apply_prefix(text: str, prefix: str) -> str:
        clean = text.strip()
        if not prefix:
            return clean

        normalized_prefix = prefix if prefix.endswith(" ") else f"{prefix} "
        if clean.lower().startswith(prefix.lower()) or clean.lower().startswith(
            normalized_prefix.lower()
        ):
            return clean
        return f"{normalized_prefix}{clean}"

    def _embed_sync(self, texts: list[str], *, prefix: str) -> list[list[float]]:
        prepared = [self._apply_prefix(text, prefix) for text in texts]
        vectors = self._model.encode(
            prepared,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts, prefix=self._passage_prefix)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._embed_sync, [text], prefix=self._query_prefix)
        return vectors[0]
