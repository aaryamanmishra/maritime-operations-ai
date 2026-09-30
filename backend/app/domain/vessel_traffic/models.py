import math
from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class NormalizedVesselEvent(BaseModel):
    """
    Normalized internal event representing an AIS message.
    Missing fields remain None; never fabricate data.
    """
    mmsi: int = Field(..., description="9-digit Maritime Mobile Service Identity")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    sog: Optional[float] = Field(None, ge=0.0, le=102.2, description="Speed Over Ground in knots")
    cog: Optional[float] = Field(None, ge=0.0, le=360.0, description="Course Over Ground in degrees")
    heading: Optional[float] = Field(None, ge=0.0, le=360.0, description="True heading in degrees")
    nav_status: Optional[int] = Field(None, ge=0, le=15, description="AIS Navigational Status code")
    
    # Static & Voyage Attributes
    ship_name: Optional[str] = None
    imo: Optional[int] = None
    ship_type: Optional[int] = None
    destination: Optional[str] = None
    callsign: Optional[str] = None
    length: Optional[float] = None
    beam: Optional[float] = None
    draught: Optional[float] = None

    @field_validator("heading", mode="before")
    @classmethod
    def normalize_heading(cls, v):
        # AIS standard: 511 indicates heading unavailable
        if v is not None and (v == 511 or v < 0 or v > 360):
            return None
        return v

    @field_validator("imo", mode="before")
    @classmethod
    def normalize_imo(cls, v):
        if v is not None and v <= 0:
            return None
        return v

    @field_validator("ship_name", "destination", "callsign", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            cleaned = v.strip()
            return cleaned if cleaned and cleaned != "@@@@@@@@@@@@@@@@@@@@" else None
        return v


class BoundingBox(BaseModel):
    min_lat: float = Field(..., ge=-90.0, le=90.0)
    min_lon: float = Field(..., ge=-180.0, le=180.0)
    max_lat: float = Field(..., ge=-90.0, le=90.0)
    max_lon: float = Field(..., ge=-180.0, le=180.0)

    @field_validator("max_lat")
    @classmethod
    def validate_latitude_order(cls, v, info):
        min_lat = info.data.get("min_lat")
        if min_lat is not None and v < min_lat:
            raise ValueError("max_lat must be greater than or equal to min_lat")
        return v

    @field_validator("max_lon")
    @classmethod
    def validate_longitude_order(cls, v, info):
        min_lon = info.data.get("min_lon")
        if min_lon is not None and v < min_lon:
            raise ValueError("max_lon must be greater than or equal to min_lon")
        return v

    def contains(self, lat: float, lon: float) -> bool:
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)


class VesselPositionPoint(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime
    sog: Optional[float] = None
    cog: Optional[float] = None
    heading: Optional[float] = None


class VesselCurrentState(BaseModel):
    mmsi: int
    name: Optional[str] = None
    imo: Optional[int] = None
    callsign: Optional[str] = None
    ship_type: Optional[int] = None
    destination: Optional[str] = None
    length: Optional[float] = None
    beam: Optional[float] = None
    draught: Optional[float] = None
    latitude: float
    longitude: float
    sog: Optional[float] = None
    cog: Optional[float] = None
    heading: Optional[float] = None
    nav_status: Optional[int] = None
    timestamp: datetime
    updated_at: datetime


class VesselDetails(VesselCurrentState):
    recent_track: List[VesselPositionPoint] = []


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two WGS84 points in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def heading_delta_deg(h1: Optional[float], h2: Optional[float]) -> float:
    """
    Computes the shortest angular difference between two compass headings in degrees (0..180),
    correctly accounting for the 0°/360° discontinuity.
    """
    if h1 is None or h2 is None:
        return 0.0
    diff = abs(h1 - h2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def should_record_history(
    last_lat: Optional[float],
    last_lon: Optional[float],
    last_heading: Optional[float],
    last_timestamp: Optional[datetime],
    new_event: NormalizedVesselEvent,
    distance_threshold_m: float = 100.0,
    heading_threshold_deg: float = 5.0,
    time_threshold_s: int = 180,
) -> bool:
    """
    Determines whether a new position report should be persisted into historical trajectory.
    Returns True if:
      - No prior recorded state exists
      - OR displacement > distance_threshold_m
      - OR heading delta > heading_threshold_deg
      - OR elapsed time > time_threshold_s
    """
    if new_event.latitude is None or new_event.longitude is None:
        return False

    if last_lat is None or last_lon is None or last_timestamp is None:
        return True

    # 1. Check distance
    dist = haversine_distance_m(last_lat, last_lon, new_event.latitude, new_event.longitude)
    if dist >= distance_threshold_m:
        return True

    # 2. Check heading
    if last_heading is not None and new_event.heading is not None:
        h_diff = heading_delta_deg(last_heading, new_event.heading)
        if h_diff >= heading_threshold_deg:
            return True

    # 3. Check elapsed time
    elapsed = abs((new_event.timestamp - last_timestamp).total_seconds())
    if elapsed >= time_threshold_s:
        return True

    return False
