import pytest
from datetime import datetime, timezone, timedelta
import numpy as np
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.domain.sar_intelligence.models import (
    MatchStatus,
    SARJobStatus,
    SARSearchRequest,
    SARSceneSummary,
    SARDetectionItem,
)
from app.infrastructure.adapters.copernicus.georeference import SARGeoreferencer
from app.infrastructure.adapters.copernicus.preprocessing import SARPreprocessor, SARTile
from app.infrastructure.adapters.copernicus.catalog import CopernicusCatalogClient
from app.domain.sar_intelligence.inference import (
    YOLO26sVesselDetector,
    non_max_suppression_boxes,
)
from app.domain.sar_intelligence.correlation import AISSARCorrelationEngine


def test_sar_bilinear_georeferencing():
    corners = {
        "top_left": (-2.0, 51.0),
        "top_right": (0.0, 51.0),
        "bottom_right": (0.0, 50.0),
        "bottom_left": (-2.0, 50.0),
    }

    # Top-Left corner (0, 0)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_bilinear(0, 0, 1000, 1000, corners)
    assert lon == -2.0
    assert lat == 51.0

    # Bottom-Right corner (1000, 1000)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_bilinear(1000, 1000, 1000, 1000, corners)
    assert lon == 0.0
    assert lat == 50.0

    # Center (500, 500)
    lon, lat = SARGeoreferencer.pixel_to_lonlat_bilinear(500, 500, 1000, 1000, corners)
    assert lon == -1.0
    assert lat == 50.5


def test_sar_preprocessing_and_tiling():
    preprocessor = SARPreprocessor(tile_size=640, tile_overlap=64)

    # 1. Test dB normalization
    band = np.array([[0.001, 0.01], [0.1, 1.0]], dtype=np.float32)
    norm = preprocessor.normalize_sar_band(band, min_db=-30.0, max_db=0.0)
    assert norm.shape == (2, 2)
    assert norm.dtype == np.uint8
    assert norm.min() >= 0
    assert norm.max() <= 255

    # 2. Test multi-channel stacking
    vv = np.ones((512, 512), dtype=np.float32) * 0.05
    vh = np.ones((512, 512), dtype=np.float32) * 0.01
    stacked = preprocessor.prepare_multichannel_sar(vv, vh)
    assert stacked.shape == (512, 512, 3)

    # 3. Test tile generation
    scene_mock = np.zeros((1200, 1200, 3), dtype=np.uint8)
    tiles = preprocessor.generate_tiles(scene_mock)
    assert len(tiles) >= 4
    for t in tiles:
        assert t.image_array.shape == (640, 640, 3)


def test_non_max_suppression():
    candidates = [
        {"bbox": [100.0, 100.0, 140.0, 140.0], "confidence": 0.95},
        {"bbox": [105.0, 102.0, 142.0, 138.0], "confidence": 0.80},  # heavily overlapping
        {"bbox": [300.0, 300.0, 340.0, 340.0], "confidence": 0.88},  # distant
    ]
    kept = non_max_suppression_boxes(candidates, iou_threshold=0.4)
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.95
    assert kept[1]["confidence"] == 0.88


def test_sar_search_request_validation():
    now = datetime.now(timezone.utc)
    req = SARSearchRequest(
        bbox=(-2.0, 49.5, 2.5, 51.5),
        start_time=now - timedelta(days=7),
        end_time=now,
        limit=10,
    )
    assert req.limit == 10

    # Inverted time range should raise ValueError
    with pytest.raises(ValueError):
        SARSearchRequest(
            bbox=(-2.0, 49.5, 2.5, 51.5),
            start_time=now,
            end_time=now - timedelta(days=1),
        )

    # Inverted bbox should raise ValueError
    with pytest.raises(ValueError):
        SARSearchRequest(
            bbox=(2.5, 49.5, -2.0, 51.5),
            start_time=now - timedelta(days=1),
            end_time=now,
        )


def test_copernicus_stac_feature_parser():
    client = CopernicusCatalogClient()
    mock_stac_feature = {
        "id": "S1A_IW_GRDH_1SDV_20260330T061540_20260330T061605_TEST",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-2.0, 50.0], [0.0, 50.0], [0.0, 51.0], [-2.0, 51.0], [-2.0, 50.0]]],
        },
        "properties": {
            "datetime": "2026-03-30T06:15:40Z",
            "platform": "Sentinel-1A",
            "sat:orbit_state": "DESCENDING",
            "sat:relative_orbit": 120,
            "sar:polarizations": ["VV", "VH"],
            "processing:level": "LEVEL1_GRD",
        },
        "assets": {
            "thumbnail": {"href": "https://example.com/thumb.png"},
            "Product": {"href": "https://example.com/product.zip"},
        },
    }

    scene = client._parse_stac_feature(mock_stac_feature)
    assert scene is not None
    assert scene.scene_id == "S1A_IW_GRDH_1SDV_20260330T061540_20260330T061605_TEST"
    assert scene.platform == "Sentinel-1A"
    assert scene.orbit_pass == "DESCENDING"
    assert scene.polarization == "VV+VH"
    assert scene.quicklook_url == "https://example.com/thumb.png"


def test_yolo26s_model_loading_and_metadata():
    """Verify that YOLO26s loads with detect task, 1 class, and accurate Kaggle metadata."""
    detector = YOLO26sVesselDetector.get_instance()
    assert detector.model is not None, "YOLO26s model failed to load"
    assert detector.task == "detect"
    assert detector.num_classes == 1
    assert detector.class_names.get(0, "").lower() == "vessel"

    # Verify metadata fields
    meta = detector.metadata
    assert meta is not None
    assert meta.get("model_name") == "YOLO26s-SAR-Vessel"
    assert meta.get("status") == "TRAINED / OPERATIONAL"
    test_metrics = meta.get("evaluation_held_out_test", {})
    assert test_metrics.get("precision") == 0.4044
    assert test_metrics.get("recall") == 0.3152
    assert test_metrics.get("mAP50") == 0.2335
    assert test_metrics.get("mAP50_95") == 0.0471


def test_yolo26s_inference_output_schema_and_confidence():
    """Verify inference output schema, confidence thresholds, and deterministic image handling."""
    detector = YOLO26sVesselDetector.get_instance()

    # Create deterministic synthetic SAR test image with bright backscatter vessel signature
    deterministic_tile = np.zeros((640, 640, 3), dtype=np.uint8)
    # Background sea clutter
    deterministic_tile[::, ::] = 20
    # Synthetic vessel point scatterers
    deterministic_tile[300:320, 310:330] = 255

    # Run inference at low confidence to inspect raw output schema
    results = detector.predict_single_tile(deterministic_tile, confidence_threshold=0.05)
    assert isinstance(results, list)
    for r in results:
        assert "bounding_box" in r
        assert len(r["bounding_box"]) == 4
        assert "confidence" in r
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["class_id"] == 0
        assert r["class_name"] == "vessel"

    # High confidence threshold should filter out background detections
    high_conf_results = detector.predict_single_tile(deterministic_tile, confidence_threshold=0.999)
    assert len(high_conf_results) <= len(results)


def test_yolo26s_scene_tiling_and_georeferencing_preservation():
    """Verify end-to-end scene tiling, coordinate remapping, and georeferencing."""
    detector = YOLO26sVesselDetector.get_instance()

    tile_img = np.full((640, 640, 3), 30, dtype=np.uint8)
    tile_img[200:220, 200:220] = 250

    tile = SARTile(
        tile_index=1,
        image_array=tile_img,
        x_offset=100,
        y_offset=150,
        width=640,
        height=640,
        parent_width=1000,
        parent_height=1000,
    )

    corners = {
        "top_left": (-2.0, 51.0),
        "top_right": (0.0, 51.0),
        "bottom_right": (0.0, 50.0),
        "bottom_left": (-2.0, 50.0),
    }

    detections = detector.predict_scene_tiles(
        tiles=[tile],
        corner_coords=corners,
        parent_width=1000,
        parent_height=1000,
        confidence_threshold=0.01,
    )

    assert isinstance(detections, list)
    for det in detections:
        assert det["detection_id"].startswith("sar-det-")
        assert 50.0 <= det["latitude"] <= 51.0
        assert -2.0 <= det["longitude"] <= 0.0
        assert "pixel_bbox" in det
        assert "source_tile" in det
        assert det["source_tile"]["tile_index"] == 1
        assert det["class_id"] == 0
        assert det["class_name"] == "vessel"


def test_ais_sar_correlation_logic():
    """Verify correlation thresholding (±15 min, 3.0 km) and match status assignment."""
    engine = AISSARCorrelationEngine(time_window_minutes=15.0, max_distance_km=3.0)

    # Case 1: Match within 1.2 km and 5 minutes
    now = datetime.now(timezone.utc)
    match_status, explanation = engine.evaluate_candidate(
        distance_km=1.2,
        time_diff_seconds=300.0,
    )
    assert match_status == MatchStatus.MATCHED
    assert "Correlated within" in explanation

    # Case 2: Unmatched due to distance (> 3.0 km)
    unmatched_status, explanation = engine.evaluate_candidate(
        distance_km=5.4,
        time_diff_seconds=300.0,
    )
    assert unmatched_status == MatchStatus.UNMATCHED
    assert "exceeded distance threshold" in explanation

    # Case 3: Unmatched due to time window (> 15 min / 900s)
    unmatched_time_status, explanation = engine.evaluate_candidate(
        distance_km=1.0,
        time_diff_seconds=1200.0,
    )
    assert unmatched_time_status == MatchStatus.UNMATCHED
    assert "exceeded time window" in explanation


@pytest.mark.asyncio
async def test_sar_api_model_status():
    """Verify model status endpoint reports TRAINED / OPERATIONAL and exact test metrics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/sar/model/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "TRAINED / OPERATIONAL"
        assert "Not certified" in data["operational_declaration"]
        assert data["task"] == "detect"

        eval_metrics = data["evaluation_held_out_test"]
        assert eval_metrics["precision"] == 0.4044
        assert eval_metrics["recall"] == 0.3152
        assert eval_metrics["mAP50"] == 0.2335
        assert eval_metrics["mAP50_95"] == 0.0471

        assert "limitations" in data
        assert "OpenSAR" in data["limitations"]


@pytest.mark.asyncio
async def test_sar_api_scenes_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/sar/scenes")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
