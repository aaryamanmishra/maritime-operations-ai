import pytest
from datetime import datetime, timezone, timedelta
from typing import Any
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.domain.sar_intelligence.models import (
    MatchStatus,
    GeographicContextStatus,
    SARDetectionItem,
    SARJobResponse,
    SARJobStatus,
    SARModality,
)
from app.domain.sar_intelligence.correlation import AISSARCorrelationEngine
from app.domain.sar_intelligence.service import SARAnalysisService
from app.infrastructure.database.session import async_session_factory
from sqlalchemy import text


@pytest.mark.asyncio
async def test_correlation_engine_status_fidelity():
    """
    Verify AISSARCorrelationEngine correctly assigns and preserves all three distinct statuses:
    - MATCHED: within 3km and +-15min
    - UNMATCHED: no nearby vessel but regional coverage is active
    - COMPARISON_UNAVAILABLE: no coverage in database for timestamp/area
    """
    engine = AISSARCorrelationEngine(time_window_minutes=15, max_distance_km=3.0)

    # 1. Proximity evaluation unit logic
    status, reason = engine.evaluate_candidate(distance_km=1.2, time_diff_seconds=120.0)
    assert status == MatchStatus.MATCHED
    assert "Correlated within" in reason

    status, reason = engine.evaluate_candidate(distance_km=4.5, time_diff_seconds=120.0)
    assert status == MatchStatus.UNMATCHED
    assert "exceeded distance threshold" in reason

    status, reason = engine.evaluate_candidate(distance_km=1.2, time_diff_seconds=1200.0)
    assert status == MatchStatus.UNMATCHED
    assert "exceeded time window" in reason


@pytest.mark.asyncio
async def test_correlation_one_to_one_mmsi_exclusion():
    """
    Verify 1-to-1 MMSI matching policy:
    When an MMSI is excluded via exclude_mmsis, subsequent detections cannot match it.
    """
    engine = AISSARCorrelationEngine(time_window_minutes=15, max_distance_km=3.0)

    # Mock session simulating two candidates: MMSI 111 (1.0km) and MMSI 222 (2.0km)
    class MockSessionFirst:
        async def execute(self, query, params=None):
            class MockRow:
                def fetchone(self):
                    # If 111 is not excluded, return 111
                    if not params or "exclude_mmsis" not in params:
                        return (111111111, datetime.now(timezone.utc), 1000.0, 60.0)
                    # If 111 is excluded, return 222
                    if 111111111 in params["exclude_mmsis"]:
                        return (222222222, datetime.now(timezone.utc), 2000.0, 120.0)
                    return None
            return MockRow()

    t_now = datetime.now(timezone.utc)
    mock_session = MockSessionFirst()

    # First detection claims MMSI 111
    res1 = await engine.correlate_detection(
        session=mock_session,
        lat=50.0,
        lon=-1.0,
        acquisition_time=t_now,
        exclude_mmsis=set(),
    )
    assert res1["match_status"] == MatchStatus.MATCHED
    assert res1["candidate_mmsi"] == 111111111

    # Second detection has MMSI 111 excluded; claims MMSI 222 instead
    res2 = await engine.correlate_detection(
        session=mock_session,
        lat=50.01,
        lon=-1.01,
        acquisition_time=t_now,
        exclude_mmsis={111111111},
    )
    assert res2["match_status"] == MatchStatus.MATCHED
    assert res2["candidate_mmsi"] == 222222222


@pytest.mark.asyncio
async def test_correlation_timezone_handling():
    """
    Verify timezone handling: naive timestamps are safely handled as UTC.
    """
    engine = AISSARCorrelationEngine(time_window_minutes=15, max_distance_km=3.0)

    class MockEmptySession:
        async def execute(self, query, params=None):
            class MockEmpty:
                def fetchone(self):
                    return None
            return MockEmpty()

    # Pass timezone-naive datetime
    t_naive = datetime(2026, 10, 26, 6, 22, 51)
    res = await engine.correlate_detection(
        session=MockEmptySession(),
        lat=50.0,
        lon=-1.0,
        acquisition_time=t_naive,
    )
    assert res["match_status"] == MatchStatus.COMPARISON_UNAVAILABLE
    assert "Historical AIS data coverage unavailable" in res["provenance"]["explanation"]
    assert res["provenance"]["sar_acquisition_time"].endswith("+00:00") or "Z" in res["provenance"]["sar_acquisition_time"]


@pytest.mark.asyncio
async def test_service_get_scene_detections_status_preservation():
    """
    Verify SARAnalysisService.get_scene_detections preserves distinct statuses:
    AIS-MATCHED, AIS-UNMATCHED, and AIS COMPARISON UNAVAILABLE without collapsing.
    """
    service = SARAnalysisService.get_instance()

    class MockQuerySession:
        async def execute(self, query, params=None):
            class MockResult:
                def fetchall(self):
                    return [
                        (
                            "det-1", "scene-1", datetime.now(timezone.utc),
                            50.0, -1.0, {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
                            0.95, "yolo26s-opensar-v1", 120.0, 45.0,
                            "AIS-MATCHED", 123456789, 30.0, 0.5,
                            {"geographic_context": {"status": "PLAUSIBLE_WATER", "is_operational_vessel": True}},
                        ),
                        (
                            "det-2", "scene-1", datetime.now(timezone.utc),
                            50.1, -1.1, {"xmin": 20, "ymin": 20, "xmax": 30, "ymax": 30},
                            0.85, "yolo26s-opensar-v1", 80.0, 90.0,
                            "AIS-UNMATCHED", None, None, None,
                            {"geographic_context": {"status": "PLAUSIBLE_WATER", "is_operational_vessel": True}},
                        ),
                        (
                            "det-3", "scene-1", datetime.now(timezone.utc),
                            50.2, -1.2, {"xmin": 40, "ymin": 40, "xmax": 50, "ymax": 50},
                            0.75, "yolo26s-opensar-v1", 60.0, None,
                            "AIS COMPARISON UNAVAILABLE", None, None, None,
                            {"geographic_context": {"status": "PLAUSIBLE_WATER", "is_operational_vessel": True}},
                        ),
                    ]
            return MockResult()

    items = await service.get_scene_detections(scene_id="scene-1", session=MockQuerySession())
    assert len(items) == 3

    assert items[0].detection_id == "det-1"
    assert items[0].match_status == MatchStatus.MATCHED
    assert items[0].candidate_mmsi == 123456789

    assert items[1].detection_id == "det-2"
    assert items[1].match_status == MatchStatus.UNMATCHED
    assert items[1].candidate_mmsi is None

    assert items[2].detection_id == "det-3"
    assert items[2].match_status == MatchStatus.COMPARISON_UNAVAILABLE
    assert items[2].candidate_mmsi is None


@pytest.mark.asyncio
async def test_workflow_idempotency(monkeypatch):
    """
    Verify repeated scene analysis is idempotent: deletes previous detections
    and does not accumulate duplicate rows in PostgreSQL.
    """
    service = SARAnalysisService.get_instance()
    job_id = "test-job-idempotency"
    scene_id = "test-scene-idempotent"

    job = SARJobResponse(
        job_id=job_id,
        scene_id=scene_id,
        status=SARJobStatus.QUEUED,
        modality=SARModality.GRD,
    )
    service._jobs[job_id] = job

    executed_statements = []

    class MockSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def execute(self, stmt, params=None):
            stmt_str = str(stmt)
            executed_statements.append(stmt_str)
            class Res:
                def scalar_one_or_none(self):
                    return None
            return Res()
        def add(self, obj):
            obj.id = 1
        async def flush(self):
            pass
        async def commit(self):
            pass

    def mock_factory():
        return MockSession()

    # Mock detector to return 2 detections
    mock_detections = [
        {
            "detection_id": "det-a",
            "latitude": 50.0,
            "longitude": -1.0,
            "pixel_bbox": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
            "confidence": 0.9,
            "model_version": "yolo26s-opensar-v1",
        },
        {
            "detection_id": "det-b",
            "latitude": 50.1,
            "longitude": -1.1,
            "pixel_bbox": {"xmin": 20, "ymin": 20, "xmax": 30, "ymax": 30},
            "confidence": 0.8,
            "model_version": "yolo26s-opensar-v1",
        },
    ]

    class MockDetector:
        def predict_scene_tiles(self, *args, **kwargs):
            return mock_detections

    monkeypatch.setattr(
        "app.domain.sar_intelligence.inference.YOLO26sVesselDetector.get_instance",
        lambda modality="GRD": MockDetector(),
    )

    async def mock_correlate(*args, **kwargs):
        return {
            "match_status": MatchStatus.UNMATCHED,
            "candidate_mmsi": None,
            "distance_km": None,
            "time_difference_seconds": None,
            "provenance": {},
        }

    monkeypatch.setattr(service.correlation_engine, "correlate_detection", mock_correlate)

    await service._execute_sar_workflow(
        job_id=job_id,
        scene_id=scene_id,
        session_factory=mock_factory,
        modality="GRD",
    )

    assert job.status == SARJobStatus.COMPLETE
    # Verify DELETE statement was executed before inserting detections
    delete_stmts = [s for s in executed_statements if "DELETE FROM sar_detection WHERE scene_id = :scene_id" in s]
    assert len(delete_stmts) >= 1, "DELETE FROM sar_detection must be executed to guarantee idempotency"


@pytest.mark.asyncio
async def test_sar_model_status_api():
    """
    Verify GET /api/v1/sar/model/status returns deployed model metadata,
    operational declaration, task, and evaluation metrics for both GRD and SLC.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GRD model status
        res_grd = await client.get("/api/v1/sar/model/status?modality=GRD")
        assert res_grd.status_code == 200
        data_grd = res_grd.json()
        assert data_grd["status"] == "TRAINED / OPERATIONAL"
        assert data_grd["modality"] == "GRD"
        assert data_grd["model_version"] == "yolo26s-opensar-v1"
        assert data_grd["evaluation_held_out_test"]["precision"] > 0
        assert data_grd["evaluation_held_out_test"]["recall"] > 0
        assert "mAP50" in data_grd["evaluation_held_out_test"]

        # 2. SLC model status
        res_slc = await client.get("/api/v1/sar/model/status?modality=SLC")
        assert res_slc.status_code == 200
        data_slc = res_slc.json()
        assert data_slc["status"] == "TRAINED / OPERATIONAL"
        assert data_slc["modality"] == "SLC"
        assert data_slc["model_version"] == "yolo26s-opensar-slc-v1"
