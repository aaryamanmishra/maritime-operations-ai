"""
Automated Test Suite for Geographic and Raster Coordinate Normalization,
Validation, Antimeridian Crossing, and Georeferencing Accuracy.
"""

import math
import pytest
import numpy as np

from app.domain.sar_intelligence.coordinates import (
    CoordinateValidationError,
    GeoPoint,
    PixelPoint,
    dead_reckon_position,
    haversine_distance_m,
    interpolate_position,
    normalize_longitude,
)
from app.infrastructure.adapters.copernicus.georeference import SARGeoreferencer
from app.infrastructure.adapters.copernicus.preprocessing import SARTile, SARPreprocessor


def test_valid_geopoint_and_conversions():
    """Verify GeoPoint accepts valid coordinates and formats for GeoJSON, Leaflet, and PostGIS."""
    pt = GeoPoint(latitude=50.5, longitude=-1.5)
    assert pt.latitude == 50.5
    assert pt.longitude == -1.5
    assert pt.to_tuple_lat_lon() == (50.5, -1.5)
    # GeoJSON & PostGIS: [lon, lat] (X, Y)
    assert pt.to_geojson_coord() == (-1.5, 50.5)
    # Leaflet: [lat, lon]
    assert pt.to_leaflet_coord() == (50.5, -1.5)
    assert pt.to_wkt_point() == "SRID=4326;POINT(-1.50000000 50.50000000)"


def test_coordinate_out_of_bounds_rejection():
    """Verify rejection of coordinates outside valid WGS84 bounds."""
    # Latitude > 90
    with pytest.raises(CoordinateValidationError, match="Latitude must be in"):
        GeoPoint(latitude=90.001, longitude=0.0)

    # Latitude < -90
    with pytest.raises(CoordinateValidationError, match="Latitude must be in"):
        GeoPoint(latitude=-90.001, longitude=0.0)

    # Non-numeric latitude
    with pytest.raises(CoordinateValidationError, match="must be a real number"):
        GeoPoint(latitude="50.5", longitude=0.0)  # type: ignore


def test_coordinate_nan_infinity_rejection():
    """Verify strict rejection of NaN and Infinity values."""
    with pytest.raises(CoordinateValidationError, match="must be finite"):
        GeoPoint(latitude=float("nan"), longitude=0.0)

    with pytest.raises(CoordinateValidationError, match="must be finite"):
        GeoPoint(latitude=float("inf"), longitude=0.0)

    with pytest.raises(CoordinateValidationError, match="must be finite"):
        GeoPoint(latitude=0.0, longitude=float("-inf"))

    with pytest.raises(CoordinateValidationError, match="must be finite"):
        GeoPoint(latitude=0.0, longitude=float("nan"))


def test_longitude_normalization_and_antimeridian():
    """Verify longitude wrapping into [-180, 180) across antimeridian."""
    # Normal in-bounds
    assert normalize_longitude(0.0) == 0.0
    assert normalize_longitude(-179.5) == -179.5
    assert normalize_longitude(179.5) == 179.5

    # Positive wraparound: 181 -> -179
    assert math.isclose(normalize_longitude(181.0), -179.0, abs_tol=1e-9)

    # Multi-turn wraparound: 540 -> -180, 720 -> 0
    assert math.isclose(normalize_longitude(541.0), -179.0, abs_tol=1e-9)
    assert math.isclose(normalize_longitude(720.0), 0.0, abs_tol=1e-9)

    # Negative wraparound: -181 -> 179
    assert math.isclose(normalize_longitude(-181.0), 179.0, abs_tol=1e-9)


def test_haversine_distance_and_antimeridian():
    """
    Verify geodesic Haversine distance correctly calculates shortest distance
    across the international antimeridian (+179.9 to -179.9).
    """
    # Points 0.2 degrees apart straddling antimeridian at equator
    # 0.2 deg at equator = ~22.24 km
    d_m = haversine_distance_m(lat1=0.0, lon1=179.9, lat2=0.0, lon2=-179.9)
    expected_m = (0.2 / 360.0) * (2.0 * math.pi * 6371008.8)
    assert math.isclose(d_m, expected_m, rel_tol=1e-4)
    assert d_m < 25000.0  # Must NOT go the long way around Earth (~40,000 km)


def test_interpolation_across_antimeridian():
    """Verify interpolation across antimeridian traverses the short path."""
    # Midpoint between +179.0 and -179.0 at equator is exactly 180.0 (or -180.0)
    mid = interpolate_position(lat1=0.0, lon1=179.0, lat2=0.0, lon2=-179.0, fraction=0.5)
    assert math.isclose(mid.latitude, 0.0, abs_tol=1e-9)
    assert abs(mid.longitude) == 180.0 or math.isclose(abs(mid.longitude), 180.0, abs_tol=1e-6)


def test_dead_reckon_position():
    """Verify dead reckoning with speed and course."""
    # Vessel at (0.0, 0.0) steaming North (course 0 deg) at 20 knots for 1 hour (3600s)
    # 20 knots * 1 hr = 20 nautical miles = 37.04 km
    proj = dead_reckon_position(lat=0.0, lon=0.0, speed_knots=20.0, course_deg=0.0, elapsed_seconds=3600.0)
    assert proj.longitude == 0.0
    # 37,040 m / 111,195 m/deg lat = ~0.333 deg
    assert math.isclose(proj.latitude, 0.3331, abs_tol=0.01)

    dist_m = haversine_distance_m(0.0, 0.0, proj.latitude, proj.longitude)
    assert math.isclose(dist_m, 20.0 * 1852.0, rel_tol=0.01)


def test_pixel_point_validation():
    """Verify PixelPoint enforces positive coordinates and raster bounds."""
    # Valid
    pt = PixelPoint(pixel_x=100.0, pixel_y=200.0, raster_width=1000.0, raster_height=1000.0)
    assert pt.pixel_x == 100.0
    assert pt.pixel_y == 200.0

    # Negative X
    with pytest.raises(CoordinateValidationError, match="cannot be negative"):
        PixelPoint(pixel_x=-1.0, pixel_y=10.0)

    # Exceed width
    with pytest.raises(CoordinateValidationError, match="exceeds raster width"):
        PixelPoint(pixel_x=1500.0, pixel_y=100.0, raster_width=1000.0, raster_height=1000.0)


def test_real_sentinel1_reference_points_error_budget():
    """
    Verify georeferencing against known reference GCP grid points from Sentinel-1 GRD.
    Measures geodesic errors in meters across corners and interior.
    """
    # Sample real GCP grid from Sentinel-1 GRD scene
    gcps = [
        {"pixel": 0.0, "line": 0.0, "longitude": -2.0, "latitude": 51.0, "height": 0.0},
        {"pixel": 10000.0, "line": 0.0, "longitude": 0.0, "latitude": 51.0, "height": 0.0},
        {"pixel": 0.0, "line": 10000.0, "longitude": -2.0, "latitude": 49.0, "height": 0.0},
        {"pixel": 10000.0, "line": 10000.0, "longitude": 0.0, "latitude": 49.0, "height": 0.0},
        {"pixel": 5000.0, "line": 5000.0, "longitude": -1.0, "latitude": 50.0, "height": 0.0},
    ]

    scene_meta = {
        "original_width": 10000,
        "original_height": 10000,
        "geolocation_grid_points": gcps,
    }

    # Test GCP reference accuracy
    errors_m = []
    for gcp in gcps:
        px, py = gcp["pixel"], gcp["line"]
        true_lon, true_lat = gcp["longitude"], gcp["latitude"]

        pred_lon, pred_lat = SARGeoreferencer.pixel_to_lonlat_grid(
            px, py, gcps
        )
        err = haversine_distance_m(pred_lat, pred_lon, true_lat, true_lon)
        errors_m.append(err)

    # At exact GCP locations, error must be sub-meter
    max_err = max(errors_m)
    mean_err = sum(errors_m) / len(errors_m)
    median_err = sorted(errors_m)[len(errors_m)//2]

    assert max_err < 5.0, f"Max GCP error too high: {max_err:.2f}m"
    assert mean_err < 2.0, f"Mean GCP error too high: {mean_err:.2f}m"
    assert median_err < 1.0, f"Median GCP error too high: {median_err:.2f}m"
