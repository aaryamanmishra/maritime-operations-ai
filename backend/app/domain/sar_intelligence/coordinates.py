"""
Geographic and Raster Coordinate Normalization, Validation, and Conversion Module.

Defines canonical WGS84 (EPSG:4326) coordinate conventions, strict axis-ordering rules,
antimeridian crossing normalization, and geodesic error calculation for the maritime intelligence pipeline.
"""

import math
from dataclasses import dataclass
from typing import Any, Tuple, Union


class CoordinateValidationError(ValueError):
    """Raised when a coordinate is malformed, out of bounds, non-finite, or invalid."""
    pass


@dataclass(frozen=True)
class GeoPoint:
    """
    Explicit geographic point in decimal degrees WGS84 (EPSG:4326).
    Invariants:
    - -90.0 <= latitude <= 90.0
    - -180.0 <= longitude < 180.0 (normalized)
    - Both latitude and longitude are strictly finite real numbers (no NaN or Inf).
    """
    latitude: float
    longitude: float

    def __post_init__(self):
        # Validate latitude
        if not isinstance(self.latitude, (int, float)):
            raise CoordinateValidationError(f"Latitude must be a real number, got {type(self.latitude)}")
        if math.isnan(self.latitude) or math.isinf(self.latitude):
            raise CoordinateValidationError(f"Latitude must be finite, got {self.latitude}")
        if not (-90.0 <= self.latitude <= 90.0):
            raise CoordinateValidationError(f"Latitude must be in [-90, 90], got {self.latitude}")

        # Validate longitude
        if not isinstance(self.longitude, (int, float)):
            raise CoordinateValidationError(f"Longitude must be a real number, got {type(self.longitude)}")
        if math.isnan(self.longitude) or math.isinf(self.longitude):
            raise CoordinateValidationError(f"Longitude must be finite, got {self.longitude}")

        # Normalize longitude to [-180.0, 180.0)
        norm_lon = normalize_longitude(self.longitude)
        object.__setattr__(self, "longitude", norm_lon)

    def to_tuple_lat_lon(self) -> Tuple[float, float]:
        """Application convention: (latitude, longitude)"""
        return (self.latitude, self.longitude)

    def to_geojson_coord(self) -> Tuple[float, float]:
        """GeoJSON / PostGIS convention: (longitude, latitude) / (X, Y)"""
        return (self.longitude, self.latitude)

    def to_leaflet_coord(self) -> Tuple[float, float]:
        """Leaflet convention: [latitude, longitude]"""
        return (self.latitude, self.longitude)

    def to_wkt_point(self) -> str:
        """PostGIS EWKT POINT with EPSG:4326"""
        return f"SRID=4326;POINT({self.longitude:.8f} {self.latitude:.8f})"


@dataclass(frozen=True)
class PixelPoint:
    """
    Raster pixel coordinate within a 2D image matrix.
    Pixel origin (0, 0) is the top-left corner of the parent raster.
    """
    pixel_x: float
    pixel_y: float
    raster_width: float | None = None
    raster_height: float | None = None

    def __post_init__(self):
        if not isinstance(self.pixel_x, (int, float)) or not isinstance(self.pixel_y, (int, float)):
            raise CoordinateValidationError("Pixel coordinates must be numeric")
        if math.isnan(self.pixel_x) or math.isinf(self.pixel_x) or math.isnan(self.pixel_y) or math.isinf(self.pixel_y):
            raise CoordinateValidationError("Pixel coordinates must be finite")
        if self.pixel_x < 0.0 or self.pixel_y < 0.0:
            raise CoordinateValidationError(f"Pixel coordinates cannot be negative: ({self.pixel_x}, {self.pixel_y})")
        if self.raster_width is not None and self.pixel_x > self.raster_width:
            raise CoordinateValidationError(f"Pixel X ({self.pixel_x}) exceeds raster width ({self.raster_width})")
        if self.raster_height is not None and self.pixel_y > self.raster_height:
            raise CoordinateValidationError(f"Pixel Y ({self.pixel_y}) exceeds raster height ({self.raster_height})")


def normalize_longitude(lon: float) -> float:
    """
    Normalize any longitude in degrees into the standard interval [-180.0, 180.0).
    Properly handles arbitrary angle multiples and antimeridian wrapping.
    """
    if math.isnan(lon) or math.isinf(lon):
        raise CoordinateValidationError(f"Cannot normalize non-finite longitude: {lon}")
    # Wrap to [-180, 180)
    norm = (lon + 180.0) % 360.0 - 180.0
    # Handle floating point negative zero or edge cases
    if norm == -180.0 and lon >= 0:
        return 180.0 - 1e-12
    return norm


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute geodesic great-circle distance between two WGS84 points in meters.
    Correctly accounts for antimeridian crossing by computing the shortest angular delta.
    Earth radius R = 6,371,008.8 meters (IUGG mean Earth radius).
    """
    R = 6371008.8

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)

    # Compute shortest delta lambda across antimeridian
    d_lon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    delta_lambda = math.radians(d_lon)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def interpolate_position(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    fraction: float,
) -> GeoPoint:
    """
    Interpolate a geographic position along the geodesic path between two points.
    fraction in [0.0, 1.0]. Handles antimeridian crossing cleanly.
    """
    fraction = min(1.0, max(0.0, fraction))
    interp_lat = lat1 + fraction * (lat2 - lat1)

    # Shortest angular distance for longitude
    delta_lon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    interp_lon = normalize_longitude(lon1 + fraction * delta_lon)

    return GeoPoint(latitude=interp_lat, longitude=interp_lon)


def dead_reckon_position(
    lat: float,
    lon: float,
    speed_knots: float,
    course_deg: float,
    elapsed_seconds: float,
) -> GeoPoint:
    """
    Dead-reckon a vessel's position after elapsed_seconds using speed and course.
    1 knot = 0.514444 m/s.
    """
    speed_mps = speed_knots * 0.51444444
    distance_m = speed_mps * elapsed_seconds
    R = 6371008.8

    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    theta = math.radians(course_deg)
    delta = distance_m / R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )

    return GeoPoint(
        latitude=math.degrees(phi2),
        longitude=normalize_longitude(math.degrees(lambda2)),
    )
