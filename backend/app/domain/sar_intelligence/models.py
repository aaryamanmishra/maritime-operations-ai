from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class MatchStatus(str, Enum):
    MATCHED = "AIS-MATCHED"
    UNMATCHED = "AIS-UNMATCHED"


class SARJobStatus(str, Enum):
    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    PREPROCESSING = "PREPROCESSING"
    INFERENCE = "INFERENCE"
    CORRELATING = "CORRELATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class SARSearchRequest(BaseModel):
    bbox: Tuple[float, float, float, float] = Field(
        ...,
        description="Spatial bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84 EPSG:4326",
    )
    start_time: datetime = Field(..., description="Start of temporal observation search window")
    end_time: datetime = Field(..., description="End of temporal observation search window")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of scenes to return")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            raise ValueError(f"Longitude must be in [-180, 180], got [{min_lon}, {max_lon}]")
        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            raise ValueError(f"Latitude must be in [-90, 90], got [{min_lat}, {max_lat}]")
        if min_lon > max_lon:
            raise ValueError(f"min_lon ({min_lon}) cannot be greater than max_lon ({max_lon})")
        if min_lat > max_lat:
            raise ValueError(f"min_lat ({min_lat}) cannot be greater than max_lat ({max_lat})")
        return v

    @field_validator("end_time")
    @classmethod
    def validate_timerange(cls, v: datetime, info: Any) -> datetime:
        if "start_time" in info.data and v <= info.data["start_time"]:
            raise ValueError("end_time must be strictly after start_time")
        return v


class SARSceneSummary(BaseModel):
    scene_id: str
    acquisition_time: datetime
    platform: str  # e.g., "Sentinel-1A"
    orbit_pass: Optional[str] = None  # "ASCENDING" | "DESCENDING"
    orbit_number: Optional[int] = None
    footprint: Dict[str, Any]  # GeoJSON Polygon geometry
    polarization: str = "VV+VH"
    processing_level: str = "LEVEL1_GRD"
    source: str = "Copernicus CDSE"
    quicklook_url: Optional[str] = None
    download_url: Optional[str] = None
    scene_metadata: Dict[str, Any] = Field(default_factory=dict)


class SARDetectionItem(BaseModel):
    detection_id: str
    scene_id: str
    acquisition_time: datetime
    latitude: float
    longitude: float
    pixel_bbox: Optional[Dict[str, float]] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    length_m: Optional[float] = None
    heading_deg: Optional[float] = None
    match_status: MatchStatus
    candidate_mmsi: Optional[int] = None
    time_difference_seconds: Optional[float] = None
    distance_km: Optional[float] = None
    matching_method: str = "spatiotemporal_nearest"
    provenance: Optional[Dict[str, Any]] = None


class SARJobResponse(BaseModel):
    job_id: str
    scene_id: str
    status: SARJobStatus
    progress_percent: int = Field(default=0, ge=0, le=100)
    message: str = "Job initialized"
    detections_count: int = 0
    matched_count: int = 0
    unmatched_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    detections: List[SARDetectionItem] = Field(default_factory=list)
    error_detail: Optional[str] = None
