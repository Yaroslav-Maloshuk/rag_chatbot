from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from app.core.config import get_settings

settings = get_settings()


class CacheService:
    def __init__(self) -> None:
        self._redis: Redis | None = None

    async def _client(self) -> Redis | None:
        if not settings.enable_cache:
            return None
        if self._redis is None:
            self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def get_json(self, key: str) -> dict[str, Any] | None:
        client = await self._client()
        if client is None:
            return None
        try:
            value = await client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return None

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        client = await self._client()
        if client is None:
            return
        ttl = ttl_seconds or settings.cache_ttl_seconds
        try:
            await client.set(key, json.dumps(value), ex=ttl)
        except Exception:  # noqa: BLE001
            return

    async def ping(self) -> bool:
        client = await self._client()
        if client is None:
            return False
        try:
            return bool(await client.ping())
        except Exception:  # noqa: BLE001
            return False
