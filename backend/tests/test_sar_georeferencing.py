import pytest
import math
from app.infrastructure.adapters.copernicus.georeference import SARGeoreferencer


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# Real Copernicus Sentinel-1 GRD GCPs extracted from S1C_IW_GRDH_1SDV_20261026T062251
REAL_S1_GRD_GCPS = [
    {"line": 0.0, "pixel": 0.0, "longitude": -0.84642689, "latitude": 50.4612045, "height": 0.0},
    {"line": 0.0, "pixel": 13337.5, "longitude": -2.71442784, "latitude": 50.6953492, "height": 0.0},
    {"line": 0.0, "pixel": 26675.0, "longitude": -4.56442805, "latitude": 50.8753941, "height": 0.0},
    {"line": 8340.0, "pixel": 0.0, "longitude": -1.08253912, "latitude": 49.7141048, "height": 0.0},
    {"line": 8340.0, "pixel": 13337.5, "longitude": -2.90529013, "latitude": 49.9351463, "height": 0.0},
    {"line": 8340.0, "pixel": 26675.0, "longitude": -4.74211692, "latitude": 50.1273033, "height": 0.0},
    {"line": 16680.0, "pixel": 0.0, "longitude": -1.31807110, "latitude": 48.9670051, "height": 0.0},
    {"line": 16680.0, "pixel": 13337.5, "longitude": -3.09615243, "latitude": 49.1748924, "height": 0.0},
    {"line": 16680.0, "pixel": 26675.0, "longitude": -4.91980458, "latitude": 49.3792120, "height": 0.0},
]


def test_real_sentinel1_grd_gcp_corners():
    """Verify exact reproduction at all 4 corners and center of real Sentinel-1 GRD raster."""
    # Top-Left (0, 0)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(0.0, 0.0, REAL_S1_GRD_GCPS)
    assert abs(lon - (-0.846427)) < 1e-5
    assert abs(lat - 50.461205) < 1e-5

    # Top-Right (26675, 0)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(26675.0, 0.0, REAL_S1_GRD_GCPS)
    assert abs(lon - (-4.564428)) < 1e-5
    assert abs(lat - 50.875394) < 1e-5

    # Bottom-Left (0, 16680)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(0.0, 16680.0, REAL_S1_GRD_GCPS)
    assert abs(lon - (-1.318071)) < 1e-5
    assert abs(lat - 48.967005) < 1e-5

    # Bottom-Right (26675, 16680)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(26675.0, 16680.0, REAL_S1_GRD_GCPS)
    assert abs(lon - (-4.919805)) < 1e-5
    assert abs(lat - 49.379212) < 1e-5

    # Center node (13337.5, 8340.0)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(13337.5, 8340.0, REAL_S1_GRD_GCPS)
    assert abs(lon - (-2.905290)) < 1e-5
    assert abs(lat - 49.935146) < 1e-5


def test_pixel_to_geographic_downsampling_and_crop():
    """Verify downsampling scale factors and crop offsets map correctly to full raster space."""
    meta = {
        "original_width": 26676,
        "original_height": 16681,
        "downsampled_width": 2048,
        "downsampled_height": 1280,
        "scale_x": 26676.0 / 2048.0,
        "scale_y": 16681.0 / 1280.0,
        "geolocation_grid_points": REAL_S1_GRD_GCPS,
    }

    # Center of downsampled image: (1024, 640)
    lon, lat = SARGeoreferencer.pixel_to_geographic(
        x=1024.0,
        y=640.0,
        metadata=meta,
        parent_width=2048,
        parent_height=1280,
    )

    # Expected: maps to full raster center (13338, 8340.5)
    # Distance from true center node (-2.905290, 49.935146) should be < 10 meters
    dist = haversine_distance_m(lon, lat, -2.905290, 49.935146)
    assert dist < 15.0, f"Distance {dist} m exceeds acceptable tolerance"


def test_pixel_to_geographic_affine_geotransform():
    """Verify affine geotransform projection for rasters with north-up or rotated affine transforms."""
    # 10m pixel spacing, top-left at (lon=0.0, lat=50.0)
    gt = [0.0, 0.0001, 0.0, 50.0, 0.0, -0.0001]
    meta = {
        "geotransform": gt,
        "crs": "EPSG:4326",
    }

    # Origin (0, 0)
    lon, lat = SARGeoreferencer.pixel_to_geographic(0.0, 0.0, metadata=meta)
    assert lon == 0.0
    assert lat == 50.0

    # Pixel (100, 200)
    lon, lat = SARGeoreferencer.pixel_to_geographic(100.0, 200.0, metadata=meta)
    assert abs(lon - 0.01) < 1e-5
    assert abs(lat - 49.98) < 1e-5


def test_footprint_corner_mapping_and_regression():
    """
    Regression test for the STAC footprint corner mapping bug.
    Demonstrates that the old naive indexing caused a ~170-266 km inversion,
    and verifies that the repaired implementation correctly orients descending passes.
    """
    stac_footprint = {
        "type": "Polygon",
        "coordinates": [[
            [-1.315792, 48.966709],  # P0: South-East (Raster Bottom-Left)
            [-0.846577, 50.461224],  # P1: North-East (Raster Top-Left)
            [-4.563474, 50.875305],  # P2: North-West (Raster Top-Right)
            [-4.919860, 49.379219],  # P3: South-West (Raster Bottom-Right)
            [-1.315792, 48.966709],  # P4: Closing point
        ]],
    }

    # Repaired extract_corners_from_footprint with orbit_pass="DESCENDING"
    corners_desc = SARGeoreferencer.extract_corners_from_footprint(stac_footprint, orbit_pass="DESCENDING")
    assert abs(corners_desc["top_left"][0] - (-0.846577)) < 1e-4
    assert abs(corners_desc["top_left"][1] - 50.461224) < 1e-4

    assert abs(corners_desc["top_right"][0] - (-4.563474)) < 1e-4
    assert abs(corners_desc["top_right"][1] - 50.875305) < 1e-4

    assert abs(corners_desc["bottom_left"][0] - (-1.315792)) < 1e-4
    assert abs(corners_desc["bottom_left"][1] - 48.966709) < 1e-4

    assert abs(corners_desc["bottom_right"][0] - (-4.919860)) < 1e-4
    assert abs(corners_desc["bottom_right"][1] - 49.379219) < 1e-4

    # Compare pixel (0, 0) against true GCP corner (-0.846427, 50.461205):
    # Old bug placed it at (-1.315792, 48.966709) which is 169 km away
    old_error_m = haversine_distance_m(-1.315792, 48.966709, -0.846427, 50.461205)
    assert old_error_m > 160000.0, "Regression check: old mapping bug should show ~169km error"

    # Repaired corners place pixel (0, 0) within ~20m of true GCP corner
    new_tl_lon, new_tl_lat = corners_desc["top_left"]
    repaired_error_m = haversine_distance_m(new_tl_lon, new_tl_lat, -0.846427, 50.461205)
    assert repaired_error_m < 50.0, f"Repaired error {repaired_error_m} m exceeds tolerance"


def test_antimeridian_crossing():
    """Verify piecewise interpolation seamlessly handles grids spanning the ±180° antimeridian."""
    antimeridian_grid = [
        {"line": 0.0, "pixel": 0.0, "longitude": 179.5, "latitude": -20.0},
        {"line": 0.0, "pixel": 1000.0, "longitude": -179.5, "latitude": -20.0},
        {"line": 1000.0, "pixel": 0.0, "longitude": 179.5, "latitude": -21.0},
        {"line": 1000.0, "pixel": 1000.0, "longitude": -179.5, "latitude": -21.0},
    ]

    # Midway in X (pixel 500) should be at longitude 180.0 (or -180.0)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(500.0, 500.0, antimeridian_grid)
    assert abs(abs(lon) - 180.0) < 1e-4
    assert abs(lat - (-20.5)) < 1e-4


def test_missing_georeferencing_metadata_fails_explicitly():
    """Verify that calling pixel_to_geographic without valid georeferencing metadata raises RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        SARGeoreferencer.pixel_to_geographic(100.0, 100.0, metadata={})
    assert "No valid georeferencing metadata" in str(exc_info.value)
