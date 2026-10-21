import math
from datetime import datetime, timezone, timedelta
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.domain.sar_intelligence.models import (
    SARModality,
    SARSearchRequest,
)
from app.infrastructure.adapters.copernicus.georeference import SARGeoreferencer
from app.infrastructure.adapters.copernicus.preprocessing import (
    SARPreprocessor,
    SLCSARPreprocessor,
    SARTile,
)
from app.infrastructure.adapters.copernicus.catalog import CopernicusCatalogClient
from app.domain.sar_intelligence.inference import YOLO26sVesselDetector


def test_slc_complex_magnitude_preprocessing():
    """Verify SLC preprocessor computes complex amplitude sqrt(I² + Q²) and percentile contrast stretch."""
    preprocessor = SLCSARPreprocessor(tile_size=640, tile_overlap=64)

    # 1. Synthetic complex-valued radar array (I + jQ)
    # Target pixel with high magnitude (e.g. real=30.0, imag=40.0 -> mag=50.0)
    real_part = np.full((100, 100), 3.0, dtype=np.float32)
    imag_part = np.full((100, 100), 4.0, dtype=np.float32)
    complex_band = real_part + 1j * imag_part
    # Insert high-intensity scatterer: real=30, imag=40 -> mag=50
    complex_band[50, 50] = 30.0 + 40.0j

    norm = preprocessor.normalize_slc_band(complex_band, p_min=2.0, p_max=99.5)
    assert norm.shape == (100, 100)
    assert norm.dtype == np.uint8
    # Background (mag=5.0) should be near zero, while scatterer (mag=50.0) should be 255
    assert norm[50, 50] == 255
    assert norm[0, 0] <= 5

    # 2. Test decode_complex_slc_bytes from raw complex64 buffer
    c_bytes = complex_band[:16, :16].astype(np.complex64).tobytes()
    decoded = preprocessor.decode_complex_slc_bytes(c_bytes, width=16, height=16)
    assert decoded.shape == (16, 16)
    assert decoded.dtype == np.uint8


def test_slc_vs_grd_preprocessing_separation():
    """Verify strict algorithmic separation between GRD (dB log-scale) and SLC (percentile magnitude)."""
    grd_proc = SARPreprocessor(tile_size=640, tile_overlap=64)
    slc_proc = SLCSARPreprocessor(tile_size=640, tile_overlap=64)

    # Test linear intensity array in [0.01, 1.0]
    test_intensity = np.linspace(0.01, 1.0, 100, dtype=np.float32).reshape(10, 10)

    # GRD uses 10 * log10(x) mapped from [-30 dB, 0 dB] to [0, 255]
    grd_norm = grd_proc.normalize_sar_band(test_intensity, min_db=-30.0, max_db=0.0)

    # SLC uses linear percentile stretch [p2, p99.5] to [0, 255]
    slc_norm = slc_proc.normalize_slc_band(test_intensity, p_min=2.0, p_max=99.5)

    # The resulting normalized pixel values must differ due to logarithmic vs. linear transfer curves
    assert not np.array_equal(grd_norm, slc_norm)

    # Check tile modality metadata tagging
    scene_mock = np.zeros((640, 640, 3), dtype=np.uint8)
    grd_tiles = grd_proc.generate_tiles(scene_mock, modality="GRD")
    slc_tiles = slc_proc.generate_tiles(scene_mock, modality="SLC")

    assert grd_tiles[0].modality == "GRD"
    assert slc_tiles[0].modality == "SLC"


def test_slc_georeferencing_and_vessel_length_geometry():
    """Verify vessel length calculations respect physical sensor geometry differences between GRD and SLC."""
    bbox = [100.0, 100.0, 120.0, 105.0]  # dx = 20 px, dy = 5 px

    # 1. GRD nominal calculation: 10m x 10m square ground range
    # dx = 200m, dy = 50m -> diag = sqrt(200^2 + 50^2) = sqrt(42500) ~= 206.2m
    len_grd = SARGeoreferencer.calculate_vessel_length(bbox, modality="GRD")
    expected_grd = round(math.sqrt((20.0 * 10.0) ** 2 + (5.0 * 10.0) ** 2), 1)
    assert len_grd == expected_grd

    # 2. SLC slant-range calculation: 3.38m slant range x 14.0m azimuth
    # dx = 20 * 3.38 = 67.6m, dy = 5 * 14.0 = 70.0m -> diag = sqrt(67.6^2 + 70.0^2) ~= 97.3m
    len_slc = SARGeoreferencer.calculate_vessel_length(bbox, modality="SLC")
    expected_slc = round(math.sqrt((20.0 * 3.38) ** 2 + (5.0 * 14.0) ** 2), 1)
    assert len_slc == expected_slc
    assert len_slc != len_grd

    # 3. SLC with custom metadata incidence angle (e.g., 35 degrees)
    meta = {"sar:incidence_angle": 35.0}
    len_slc_inc = SARGeoreferencer.calculate_vessel_length(bbox, modality="SLC", metadata=meta)
    range_ground = 3.38 / math.sin(math.radians(35.0))  # ~5.89m
    expected_inc = round(math.sqrt((20.0 * range_ground) ** 2 + (5.0 * 14.0) ** 2), 1)
    assert len_slc_inc == expected_inc


def test_modality_selection_defaults_and_opt_in():
    """Verify GRD remains default across requests and catalog search, while SLC is strictly opt-in."""
    now = datetime.now(timezone.utc)

    # 1. Default request must be GRD
    default_req = SARSearchRequest(
        bbox=(-2.0, 49.5, 2.5, 51.5),
        start_time=now - timedelta(days=2),
        end_time=now,
    )
    assert default_req.modality == SARModality.GRD
    assert default_req.modality.value == "GRD"

    # 2. Explicit opt-in request for SLC
    slc_req = SARSearchRequest(
        bbox=(-2.0, 49.5, 2.5, 51.5),
        start_time=now - timedelta(days=2),
        end_time=now,
        modality=SARModality.SLC,
    )
    assert slc_req.modality == SARModality.SLC
    assert slc_req.modality.value == "SLC"

    # 3. Catalog client targets correct collections
    client = CopernicusCatalogClient()
    # Mock payload verification
    # When modality=GRD -> target_collection='sentinel-1-grd'
    # When modality=SLC -> target_collection='sentinel-1-slc'
    feat_mock = {
        "id": "S1A_IW_SLC__1SDV_TEST",
        "geometry": {"type": "Polygon", "coordinates": [[[-2.0, 50.0], [0.0, 50.0], [0.0, 51.0], [-2.0, 51.0], [-2.0, 50.0]]]},
        "properties": {"datetime": "2026-10-20T10:00:00Z", "processing:level": "LEVEL1_SLC"},
        "assets": {"thumbnail": {"href": "https://example.com/thumb.png"}},
    }
    summary_slc = client._parse_stac_feature(feat_mock, modality="SLC")
    assert summary_slc.modality == "SLC"
    assert summary_slc.processing_level == "LEVEL1_SLC"


def test_yolo26s_detector_modality_routing():
    """Verify detector factory returns distinct detector instances for GRD and SLC checkpoints."""
    detector_grd = YOLO26sVesselDetector.get_instance(modality="GRD")
    detector_slc = YOLO26sVesselDetector.get_instance(modality="SLC")

    assert detector_grd.modality == "GRD"
    assert detector_slc.modality == "SLC"
    assert "yolo26s_sar_vessel.pt" in detector_grd.weights_path
    assert "yolo26s_slc_vessel.pt" in detector_slc.weights_path
    assert detector_grd.model_version == "yolo26s-opensar-v1"
    assert detector_slc.model_version == "yolo26s-opensar-slc-v1"


def test_slc_inference_smoke_and_output_schema():
    """Run real inference on an SLC test tile using the opt-in SLC pipeline and verify output schema."""
    import cv2
    import os

    detector_slc = YOLO26sVesselDetector.get_instance(modality="SLC")
    test_img_candidates = [
        os.path.join(os.path.dirname(__file__), "../app/domain/sar_intelligence/data/slc_test_chip.jpg"),
        "app/domain/sar_intelligence/data/slc_test_chip.jpg",
        "backend/app/domain/sar_intelligence/data/slc_test_chip.jpg",
        "data/processed/opensar_slc/images/test/DB_OPENSAR_VD_104.jpg",
        "../data/processed/opensar_slc/images/test/DB_OPENSAR_VD_104.jpg",
    ]
    test_img_path = next((p for p in test_img_candidates if os.path.exists(p)), None)
    if test_img_path is not None:
        img = cv2.imread(test_img_path)
    else:
        # Fallback synthetic array with bright scatterers for environments without chip
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[::, ::] = 20
        img[300:320, 310:330] = 255
    dets = detector_slc.predict_single_tile(img, confidence_threshold=0.25)

    assert len(dets) >= 1
    for d in dets:
        assert "bounding_box" in d
        assert len(d["bounding_box"]) == 4
        assert 0.0 <= d["confidence"] <= 1.0
        assert d["class_id"] == 0
        assert d["class_name"] == "vessel"

    # Test scene tiling and georeferencing
    tile = SARTile(
        tile_index=0,
        image_array=img,
        x_offset=100,
        y_offset=150,
        width=640,
        height=640,
        parent_width=1000,
        parent_height=1000,
        modality="SLC",
    )
    corners = {
        "top_left": (-2.0, 51.0),
        "top_right": (0.0, 51.0),
        "bottom_right": (0.0, 50.0),
        "bottom_left": (-2.0, 50.0),
    }

    scene_dets = detector_slc.predict_scene_tiles(
        tiles=[tile],
        corner_coords=corners,
        parent_width=1000,
        parent_height=1000,
        confidence_threshold=0.25,
        scene_metadata={"sar:incidence_angle": 38.5},
    )

    assert len(scene_dets) >= 1
    for sd in scene_dets:
        assert sd["detection_id"].startswith("sar-det-")
        assert sd["modality"] == "SLC"
        assert sd["model_version"] == "yolo26s-opensar-slc-v1"
        assert 50.0 <= sd["latitude"] <= 51.0
        assert -2.0 <= sd["longitude"] <= 0.0
        assert sd["length_m"] > 0


@pytest.mark.asyncio
async def test_model_status_api_modality_support():
    """Verify /api/v1/sar/model/status supports both default GRD and opt-in SLC metrics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Default request returns GRD metrics
        resp_default = await client.get("/api/v1/sar/model/status")
        assert resp_default.status_code == 200
        data_grd = resp_default.json()
        assert data_grd["modality"] == "GRD"
        assert data_grd["model_version"] == "yolo26s-opensar-v1"
        assert data_grd["evaluation_held_out_test"]["precision"] == 0.4044
        assert data_grd["evaluation_held_out_test"]["recall"] == 0.3152
        assert data_grd["evaluation_held_out_test"]["mAP50"] == 0.2335

        # 2. Opt-in request with modality=SLC returns verified SLC metrics
        resp_slc = await client.get("/api/v1/sar/model/status?modality=SLC")
        assert resp_slc.status_code == 200
        data_slc = resp_slc.json()
        assert data_slc["modality"] == "SLC"
        assert data_slc["model_version"] == "yolo26s-opensar-slc-v1"
        assert data_slc["evaluation_held_out_test"]["precision"] == 0.5255
        assert data_slc["evaluation_held_out_test"]["recall"] == 0.4164
        assert data_slc["evaluation_held_out_test"]["mAP50"] == 0.3792
        assert data_slc["evaluation_held_out_test"]["mAP50_95"] == 0.1284


@pytest.mark.asyncio
async def test_cdse_burst_search_and_query_generation(monkeypatch):
    """Verify OData Bursts search builds compliant query filters for subswaths and footprints."""
    client = CopernicusCatalogClient()

    captured_url = None
    captured_params = None

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, url, params=None):
            nonlocal captured_url, captured_params
            captured_url = url
            captured_params = params

            class MockResponse:
                status_code = 200

                def json(self):
                    return {
                        "value": [
                            {
                                "Id": "mock-burst-uuid-1",
                                "Name": "S1A-SLC-20240803T001643-039501-IW1-VV-439472",
                                "BurstId": 39501,
                                "SwathIdentifier": "IW1",
                                "LinesPerBurst": 1496,
                                "SamplesPerBurst": 21330,
                            }
                        ]
                    }

            return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: MockAsyncClient())

    results = await client.search_slc_bursts(
        bbox=(103.5, 1.1, 104.2, 1.5),
        burst_id=39501,
        swath_id="IW1",
        polarization="VV",
    )

    assert len(results) == 1
    assert results[0]["Id"] == "mock-burst-uuid-1"
    assert "Bursts" in captured_url
    assert "BurstId eq 39501" in captured_params["$filter"]
    assert "SwathIdentifier eq 'IW1'" in captured_params["$filter"]
    assert "PolarisationChannels eq 'VV'" in captured_params["$filter"]
    assert "OData.CSC.Intersects" in captured_params["$filter"]


@pytest.mark.asyncio
async def test_cdse_burst_download_otf_protocol_and_error_handling(monkeypatch, tmp_path):
    """Verify CDSE burst acquisition handles the two-step POST flow and intercepts provider OTF errors."""
    from app.infrastructure.adapters.copernicus.acquisition import CopernicusAcquisitionClient

    acq_client = CopernicusAcquisitionClient()

    # Mock get_access_token
    async def mock_token():
        return "mock-bearer-token"

    monkeypatch.setattr(acq_client, "get_access_token", mock_token)

    # 1. Simulate CDSE On-The-Fly (OTF) failure (HTTP 500 PRO-OTF-005)
    class MockFailingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def post(self, url, headers=None, **kwargs):
            class MockRedirectResponse:
                status_code = 301
                headers = {"Location": "https://bursts.dataspace.copernicus.eu/v1/product/mock-id"}

            return MockRedirectResponse()

        def stream(self, method, url, headers=None, **kwargs):
            class MockStreamContext:
                async def __aenter__(ctx):
                    class MockErrorStream:
                        status_code = 500

                        async def aread(self):
                            return b'{"detail":{"code":"PRO-OTF-005","message":"Error processing burst with GDAL"}}'

                    return MockErrorStream()

                async def __aexit__(ctx, exc_type, exc_val, exc_tb):
                    pass

            return MockStreamContext()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: MockFailingAsyncClient())

    target_zip = str(tmp_path / "burst.zip")
    with pytest.raises(RuntimeError) as exc_info:
        await acq_client.download_slc_burst("mock-id", target_path=target_zip)

    assert "PRO-OTF-005" in str(exc_info.value)
    assert "HTTP 500" in str(exc_info.value)


def test_slc_verified_iw1_pixel_spacing_and_vessel_length():
    """Verify SARGeoreferencer uses verified Sentinel-1 IW1 range pixel spacing (2.33m)."""
    bbox = [100.0, 100.0, 120.0, 105.0]  # dx = 20 px, dy = 5 px

    # 1. Nominal fallback SLC geometry (nominal 3.38m range x 14.0m azimuth)
    len_nominal = SARGeoreferencer.calculate_vessel_length(bbox, modality="SLC")
    expected_nominal = round(math.sqrt((20.0 * 3.38) ** 2 + (5.0 * 14.0) ** 2), 1)
    assert len_nominal == expected_nominal  # 97.3m slant-range

    # 2. Verified IW1 geometry with incidence angle from acquired product (theta = 30.55628 deg)
    meta = {
        "rangePixelSpacing": 2.329562,
        "azimuthPixelSpacing": 13.96549,
        "sar:subswath": "IW1",
        "sar:incidence_angle": 30.55628,
    }
    len_verified = SARGeoreferencer.calculate_vessel_length(bbox, modality="SLC", metadata=meta)

    inc_rad = math.radians(30.55628)
    ground_range = 2.329562 / math.sin(inc_rad)  # ~4.582 m
    expected_verified = round(math.sqrt((20.0 * ground_range) ** 2 + (5.0 * 13.96549) ** 2), 1)
    assert len_verified == expected_verified
    assert len_verified == 115.2  # 115.2m ground range, NOT 150.3m (flawed 3.38m assumption)


def test_slc_piecewise_gcp_grid_georeferencing():
    """Verify piecewise GCP grid interpolation reproduces product GCPs accurately without km-scale distortion."""
    grid_fixture = [
        {"line": 0.0, "pixel": 0.0, "latitude": 39.369857, "longitude": 90.638952, "incidenceAngle": 30.56},
        {"line": 0.0, "pixel": 1066.0, "latitude": 39.378088, "longitude": 90.583624, "incidenceAngle": 30.90},
        {"line": 0.0, "pixel": 2132.0, "latitude": 39.386335, "longitude": 90.527983, "incidenceAngle": 31.24},
        {"line": 1496.0, "pixel": 0.0, "latitude": 39.191696, "longitude": 90.577905, "incidenceAngle": 30.55},
        {"line": 1496.0, "pixel": 1066.0, "latitude": 39.199926, "longitude": 90.522588, "incidenceAngle": 30.89},
        {"line": 1496.0, "pixel": 2132.0, "latitude": 39.208173, "longitude": 90.466958, "incidenceAngle": 31.23},
    ]

    # 1. Exact reproduction at GCP nodes
    for p in grid_fixture:
        lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(p["pixel"], p["line"], grid_fixture)
        assert abs(lon - p["longitude"]) < 1e-5
        assert abs(lat - p["latitude"]) < 1e-5

    # 2. Interior interpolation
    mid_lon, mid_lat = SARGeoreferencer.pixel_to_lonlat_grid(533.0, 748.0, grid_fixture)
    assert 90.52 < mid_lon < 90.64
    assert 39.19 < mid_lat < 39.38


@pytest.mark.asyncio
async def test_slc_burst_acquisition_adapter_with_fixtures(tmp_path):
    """Verify CopernicusAcquisitionClient.acquire_slc_burst loads verified fixtures and metadata in test mode."""
    from app.infrastructure.adapters.copernicus.acquisition import CopernicusAcquisitionClient

    acq_client = CopernicusAcquisitionClient()
    burst_result = await acq_client.acquire_slc_burst(
        scene_id="test_scene_iw1",
        target_dir=str(tmp_path),
        allow_offline_fixtures=True,
    )

    assert burst_result["raster_path"] is not None
    assert "scene_fused.png" in burst_result["raster_path"]
    meta = burst_result["metadata"]
    assert meta["rangePixelSpacing"] == 2.329562
    assert meta["azimuthPixelSpacing"] == 13.96549
    assert meta["sar:subswath"] == "IW1"
    assert len(meta.get("geolocation_grid_points", [])) == 42


@pytest.mark.asyncio
async def test_real_s1_acquisition_fails_explicitly_when_credentials_missing(tmp_path):
    """Verify real S1 acquisitions fail with clear error when CDSE credentials are missing."""
    from app.infrastructure.adapters.copernicus.acquisition import CopernicusAcquisitionClient

    acq_client = CopernicusAcquisitionClient()
    acq_client.settings.CDSE_CLIENT_ID = None
    acq_client.settings.CDSE_CLIENT_SECRET = None
    acq_client.settings.CDSE_USERNAME = None
    acq_client.settings.CDSE_PASSWORD = None

    with pytest.raises(RuntimeError) as exc_info:
        await acq_client.acquire_slc_burst(
            scene_id="S1A_IW_SLC__1SDV_20240810T000834",
            target_dir=str(tmp_path),
            allow_offline_fixtures=True,  # Even with allow_offline_fixtures, real S1 must never use fixtures!
        )
    assert "CDSE credentials are not configured" in str(exc_info.value)
    assert "S1A_IW_SLC__1SDV_20240810T000834" in str(exc_info.value)


@pytest.mark.asyncio
async def test_real_s1_acquisition_fails_explicitly_when_gdal_unavailable(monkeypatch, tmp_path):
    """Verify real S1 acquisitions fail explicitly when gdal_translate is unavailable."""
    from app.infrastructure.adapters.copernicus.acquisition import CopernicusAcquisitionClient
    import shutil
    import os

    acq_client = CopernicusAcquisitionClient()
    monkeypatch.setattr(acq_client, "is_configured", lambda: True)
    orig_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: False if "gdal" in str(p) else orig_exists(p))
    orig_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda cmd: None if "gdal" in str(cmd) else orig_which(cmd))

    with pytest.raises(RuntimeError) as exc_info:
        await acq_client.acquire_slc_burst(
            scene_id="S1A_IW_SLC__1SDV_20240810T000834",
            target_dir=str(tmp_path),
        )
    assert "gdal_translate binary not available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unauthorized_offline_fixture_access_rejected(tmp_path):
    """Verify non-S1 scene IDs reject offline fixtures unless explicitly permitted."""
    from app.infrastructure.adapters.copernicus.acquisition import CopernicusAcquisitionClient

    acq_client = CopernicusAcquisitionClient()
    with pytest.raises(RuntimeError) as exc_info:
        await acq_client.acquire_slc_burst(
            scene_id="test_scene_iw1",
            target_dir=str(tmp_path),
            allow_offline_fixtures=False,
        )
    assert "Offline fixture usage is restricted to explicit test invocations" in str(exc_info.value)


@pytest.mark.asyncio
async def test_slc_service_workflow_end_to_end(monkeypatch):
    """Verify SARAnalysisService._execute_sar_workflow wires SLC burst acquisition, GCP grid, and provenance."""
    import json
    import os
    from app.domain.sar_intelligence.service import SARAnalysisService
    from app.domain.sar_intelligence.models import SARJobResponse, SARJobStatus, SARModality, MatchStatus

    service = SARAnalysisService.get_instance()
    job_id = "test-slc-job-1"
    job = SARJobResponse(
        job_id=job_id,
        scene_id="S1A_IW_SLC__1SDV_20240810T000834",
        status=SARJobStatus.QUEUED,
        modality=SARModality.SLC,
    )
    service._jobs[job_id] = job

    # Mock DB session factory to run without PostgreSQL
    class MockAsyncSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def execute(self, stmt):
            class MockResult:
                def scalar_one_or_none(self):
                    return None

            return MockResult()

        def add(self, obj):
            obj.id = 1

        async def flush(self):
            pass

        async def commit(self):
            pass

    def mock_session_factory():
        return MockAsyncSession()

    # Mock correlation engine
    async def mock_correlate(*args, **kwargs):
        return {
            "match_status": MatchStatus.UNMATCHED,
            "candidate_mmsi": None,
            "distance_km": None,
            "time_difference_seconds": None,
            "provenance": {"source": "mock_ais"},
        }

    monkeypatch.setattr(service.correlation_engine, "correlate_detection", mock_correlate)

    # Mock burst acquisition with verified test fixture to test service wiring
    test_chip = os.path.join(os.path.dirname(__file__), "../app/domain/sar_intelligence/data/slc_test_chip.jpg")
    meta_fixture = os.path.join(os.path.dirname(__file__), "../app/domain/sar_intelligence/data/burst_258592_metadata.json")
    with open(meta_fixture) as f:
        loaded_meta = json.load(f)
    burst_meta = {
        "rangePixelSpacing": 2.329562,
        "azimuthPixelSpacing": 13.96549,
        "sar:subswath": "IW1",
        "sar:incidence_angle": 30.55628,
        "geolocation_grid_points": loaded_meta.get("geolocation_grid_points", []),
        "acquisition_method": "mock_burst_fixture",
    }

    async def mock_acquire_slc_burst(*args, **kwargs):
        return {
            "raster_path": test_chip,
            "metadata": burst_meta,
            "burst_id": 258592,
        }

    monkeypatch.setattr(service.acquisition_client, "acquire_slc_burst", mock_acquire_slc_burst)

    await service._execute_sar_workflow(
        job_id=job_id,
        scene_id="S1A_IW_SLC__1SDV_20240810T000834",
        session_factory=mock_session_factory,
        modality="SLC",
    )

    assert job.status == SARJobStatus.COMPLETE
    assert job.modality == SARModality.SLC
    assert job.progress_percent == 100
    for det in job.detections:
        assert det.modality == "SLC"
        assert det.model_version == "yolo26s-opensar-slc-v1"
        assert det.provenance["geolocation_method"] == "piecewise_gcp_grid"
        assert det.provenance["range_pixel_spacing_m"] == 2.329562
        assert det.provenance["azimuth_pixel_spacing_m"] == 13.96549


@pytest.mark.asyncio
async def test_slc_service_workflow_fails_explicitly_on_acquisition_error(monkeypatch):
    """Verify failed SLC acquisitions set job status to FAILED and never substitute synthetic data."""
    from app.domain.sar_intelligence.service import SARAnalysisService
    from app.domain.sar_intelligence.models import SARJobResponse, SARJobStatus, SARModality

    service = SARAnalysisService.get_instance()
    job_id = "test-slc-job-fail"
    job = SARJobResponse(
        job_id=job_id,
        scene_id="S1A_IW_SLC__1SDV_20240810T000834",
        status=SARJobStatus.QUEUED,
        modality=SARModality.SLC,
    )
    service._jobs[job_id] = job

    class MockAsyncSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def execute(self, stmt):
            class MockResult:
                def scalar_one_or_none(self):
                    return None

            return MockResult()

        def add(self, obj):
            pass

        async def commit(self):
            pass

    # Simulate real SLC acquisition failure (e.g. unconfigured credentials)
    async def mock_failing_burst(*args, **kwargs):
        raise RuntimeError("CDSE credentials are not configured for Sentinel-1 SLC burst acquisition")

    monkeypatch.setattr(service.acquisition_client, "acquire_slc_burst", mock_failing_burst)

    detector_called = False

    def mock_predict(*args, **kwargs):
        nonlocal detector_called
        detector_called = True
        return []

    detector = YOLO26sVesselDetector.get_instance(modality="SLC")
    monkeypatch.setattr(detector, "predict_scene_tiles", mock_predict)

    await service._execute_sar_workflow(
        job_id=job_id,
        scene_id="S1A_IW_SLC__1SDV_20240810T000834",
        session_factory=lambda: MockAsyncSession(),
        modality="SLC",
    )

    assert job.status == SARJobStatus.FAILED
    assert "SAR analysis failed" in job.message
    assert "CDSE credentials are not configured" in job.message
    assert not detector_called, "Detector must not run on failed or synthetic data"
    assert job.detections == []


@pytest.mark.asyncio
async def test_slc_service_workflow_fails_when_no_valid_raster_returned(monkeypatch):
    """Verify that returning no valid raster path in SLC workflow triggers FAILED job status."""
    from app.domain.sar_intelligence.service import SARAnalysisService
    from app.domain.sar_intelligence.models import SARJobResponse, SARJobStatus, SARModality

    service = SARAnalysisService.get_instance()
    job_id = "test-slc-job-no-raster"
    job = SARJobResponse(
        job_id=job_id,
        scene_id="S1A_IW_SLC__1SDV_20240810T000834",
        status=SARJobStatus.QUEUED,
        modality=SARModality.SLC,
    )
    service._jobs[job_id] = job

    class MockAsyncSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def execute(self, stmt):
            class MockResult:
                def scalar_one_or_none(self):
                    return None

            return MockResult()

        def add(self, obj):
            pass

        async def commit(self):
            pass

    async def mock_empty_burst(*args, **kwargs):
        return {"raster_path": None, "metadata": {}}

    monkeypatch.setattr(service.acquisition_client, "acquire_slc_burst", mock_empty_burst)

    await service._execute_sar_workflow(
        job_id=job_id,
        scene_id="S1A_IW_SLC__1SDV_20240810T000834",
        session_factory=lambda: MockAsyncSession(),
        modality="SLC",
    )

    assert job.status == SARJobStatus.FAILED
    assert "no valid raster data" in job.message
    assert job.detections == []
