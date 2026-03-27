from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.presentation.http.deps import get_session
from app.services.factory import get_cache_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(session: AsyncSession = Depends(get_session)) -> dict[str, bool]:
    db_ok = False
    redis_ok = False

    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False

    try:
        redis_ok = await get_cache_service().ping()
    except Exception:  # noqa: BLE001
        redis_ok = False

    return {"database": db_ok, "redis": redis_ok}

