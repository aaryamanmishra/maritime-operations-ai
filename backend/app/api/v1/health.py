from datetime import datetime, timezone
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from app.infrastructure.database.session import check_database_connection
from app.infrastructure.redis.client import check_redis_connection

router = APIRouter(prefix="/health", tags=["Health"])


class LivenessResponse(BaseModel):
    status: str
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str
    database: str
    redis: str
    timestamp: str


@router.get("/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    return LivenessResponse(
        status="alive",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(response: Response) -> ReadinessResponse:
    db_ok = await check_database_connection()
    redis_ok = await check_redis_connection()

    is_ready = db_ok and redis_ok
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "degraded",
        database="ok" if db_ok else "unavailable",
        redis="ok" if redis_ok else "unavailable",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
