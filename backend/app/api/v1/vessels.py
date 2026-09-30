import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.vessel_traffic.models import (
    BoundingBox, VesselCurrentState, VesselDetails
)
from app.infrastructure.database.session import get_db_session
from app.infrastructure.database.repositories.vessel_repository import VesselRepository
from app.infrastructure.redis.client import get_redis_client

router = APIRouter(prefix="/vessels", tags=["Vessels"])


@router.get("", response_model=List[VesselCurrentState])
async def get_vessels_in_bbox(
    min_lat: float = Query(..., ge=-90.0, le=90.0, description="Minimum latitude"),
    min_lon: float = Query(..., ge=-180.0, le=180.0, description="Minimum longitude"),
    max_lat: float = Query(..., ge=-90.0, le=90.0, description="Maximum latitude"),
    max_lon: float = Query(..., ge=-180.0, le=180.0, description="Maximum longitude"),
    limit: int = Query(500, ge=1, le=1000, description="Maximum number of vessels to return"),
    session: AsyncSession = Depends(get_db_session),
) -> List[VesselCurrentState]:
    """
    Spatial query returning all current vessels within the specified bounding box.
    Uses PostGIS GiST index for fast spatial indexing.
    """
    try:
        bbox = BoundingBox(
            min_lat=min_lat,
            min_lon=min_lon,
            max_lat=max_lat,
            max_lon=max_lon,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    repo = VesselRepository(session)
    return await repo.get_vessels_in_bbox(bbox=bbox, limit=limit)


@router.get("/pipeline/status", response_model=Dict[str, Any])
async def get_pipeline_status(session: AsyncSession = Depends(get_db_session)) -> Dict[str, Any]:
    """
    Returns live AIS pipeline observability metrics and connection health.
    """
    redis = get_redis_client()
    raw_status = await redis.get(settings.AIS_STATUS_REDIS_KEY)

    metrics: Dict[str, Any] = {
        "connection_state": "disconnected",
        "messages_received": 0,
        "normalized_count": 0,
        "stream_writes": 0,
        "persisted_count": 0,
        "last_message_at": None,
        "connected_at": None,
        "bbox": [
            settings.AIS_BBOX_MIN_LAT,
            settings.AIS_BBOX_MIN_LON,
            settings.AIS_BBOX_MAX_LAT,
            settings.AIS_BBOX_MAX_LON,
        ],
    }

    if raw_status:
        try:
            metrics.update(json.loads(raw_status))
        except Exception:
            pass

    repo = VesselRepository(session)
    metrics["total_vessels_in_db"] = await repo.count_total_vessels()

    return metrics


@router.get("/{mmsi}", response_model=VesselDetails)
async def get_vessel_details(
    mmsi: int,
    history_limit: int = Query(50, ge=1, le=200, description="Max recent track points"),
    session: AsyncSession = Depends(get_db_session),
) -> VesselDetails:
    """
    Returns full vessel identity, current kinematics, and recent recorded track points.
    """
    repo = VesselRepository(session)
    vessel = await repo.get_vessel_details(mmsi=mmsi, history_limit=history_limit)
    if not vessel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vessel with MMSI {mmsi} not found",
        )
    return vessel
