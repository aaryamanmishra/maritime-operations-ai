"""
Automated Test Suite for Kinematic Uncertainty-Aware AIS-SAR Correlation,
Soft-Edge Decision Boundaries, and Dark-Vessel Semantics.
"""

import math
from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.sar_intelligence.correlation import AISSARCorrelationEngine
from app.domain.sar_intelligence.models import MatchStatus


def test_uncertainty_budget_growth():
    """Verify uncertainty budget increases realistically with elapsed time and speed."""
    engine = AISSARCorrelationEngine(time_window_minutes=30)

    # Stationary vessel (0 knots) over 60 seconds
    unc_stat = engine.compute_uncertainty_budget(speed_knots=0.0, elapsed_seconds=60.0)
    assert unc_stat["sigma_sar_m"] == 30.0
    assert unc_stat["sigma_ais_m"] == 15.0
    assert unc_stat["sigma_motion_m"] >= 15.0
    assert 30.0 < unc_stat["sigma_total_m"] < 60.0

    # Fast vessel (20 knots ~ 10.3 m/s) over 900 seconds (15 min)
    unc_fast = engine.compute_uncertainty_budget(speed_knots=20.0, elapsed_seconds=900.0)
    assert unc_fast["sigma_motion_m"] > unc_stat["sigma_motion_m"]
    assert unc_fast["sigma_total_m"] > 500.0


def test_evaluate_candidate_high_confidence_match():
    """Verify a detection within hard gate and low z-score produces MATCHED."""
    engine = AISSARCorrelationEngine(time_window_minutes=30)
    # Candidate at 50 meters, 30 seconds offset, 10 knots
    status, reason, metrics = engine.evaluate_candidate(
        distance_km=0.05,
        time_diff_seconds=30.0,
        speed_knots=10.0,
        return_metrics=True,
    )
    assert status == MatchStatus.MATCHED
    assert metrics["z_score"] < 2.5
    assert metrics["likelihood_score"] > 0.05
    assert "High-confidence correlation" in reason


def test_evaluate_candidate_soft_edge_review_required():
    """Verify candidate near the decision boundary produces DETECTION-REVIEW-REQUIRED."""
    engine = AISSARCorrelationEngine(time_window_minutes=30, max_distance_km=4.0)
    # Fast vessel where candidate is in soft-edge zone
    unc = engine.compute_uncertainty_budget(speed_knots=15.0, elapsed_seconds=300.0)
    sigma = unc["sigma_total_m"]
    # Place candidate at 3.0 sigma (outside 2.5 sigma hard gate, but within 4.0 sigma soft gate)
    test_dist_m = 3.0 * sigma
    test_dist_km = test_dist_m / 1000.0

    status, reason, metrics = engine.evaluate_candidate(
        distance_km=test_dist_km,
        time_diff_seconds=300.0,
        speed_knots=15.0,
        return_metrics=True,
    )
    assert status == MatchStatus.DETECTION_REVIEW_REQUIRED
    assert "Ambiguous soft-edge candidate" in reason
    assert 2.5 < metrics["z_score"] <= 4.0


def test_evaluate_candidate_distant_unmatched():
    """Verify candidate far outside gate produces UNMATCHED."""
    engine = AISSARCorrelationEngine(time_window_minutes=30, max_distance_km=3.0)
    status, reason, metrics = engine.evaluate_candidate(
        distance_km=15.0,
        time_diff_seconds=60.0,
        speed_knots=10.0,
        return_metrics=True,
    )
    assert status == MatchStatus.UNMATCHED
    assert "exceeded gates" in reason
    assert metrics["likelihood_score"] < 1e-6


@pytest.mark.asyncio
async def test_correlate_detection_with_dead_reckoning():
    """
    Verify kinematic dead reckoning: a vessel that was 3 km away 10 minutes ago steaming
    directly toward the SAR target at 10 knots (~3.1 km in 10 mins) correlates as MATCHED.
    """
    engine = AISSARCorrelationEngine(time_window_minutes=30)
    session = AsyncMock()

    sar_time = datetime(2026, 3, 12, 6, 0, 0, tzinfo=timezone.utc)
    # AIS report 10 minutes prior (5:50) at lat 50.0, lon -1.0, heading North (course 0 deg) at 10 knots
    ais_time = sar_time - timedelta(minutes=10)

    # 10 knots = 5.144 m/s. In 600s, vessel travels 3,086.6 meters North (~0.0277 deg lat).
    # Target SAR detection is located at the anticipated arrival position!
    sar_lat = 50.0 + (3086.6 / 111195.0)
    sar_lon = -1.0

    # Mock DB returns the raw ping 10 mins ago (which was raw_dist_m ~ 3086m)
    mock_row = (123456789, ais_time, 50.0, -1.0, 10.0, 0.0, 3086.0, 600.0)
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [mock_row]
    session.execute.return_value = mock_cursor

    res = await engine.correlate_detection(
        session=session,
        lat=sar_lat,
        lon=sar_lon,
        acquisition_time=sar_time,
    )

    assert res["match_status"] == MatchStatus.MATCHED
    assert res["candidate_mmsi"] == 123456789
    prov = res["provenance"]
    assert prov["estimated_movement_m"] > 2500.0
    # Projected distance after dead reckoning is within ~100m!
    assert prov["distance_km"] < 0.20


@pytest.mark.asyncio
async def test_dark_vessel_candidate_when_regional_coverage_active():
    """Verify an unassociated target produces DARK-VESSEL-CANDIDATE when regional coverage is verified."""
    engine = AISSARCorrelationEngine(time_window_minutes=30)
    session = AsyncMock()

    # 1. First query (nearby candidates) returns no rows
    mock_cand_cursor = MagicMock()
    mock_cand_cursor.fetchall.return_value = []

    # 2. Coverage query returns 5 active vessels in region
    mock_cov_cursor = MagicMock()
    mock_cov_cursor.fetchone.return_value = (5,)

    session.execute.side_effect = [mock_cand_cursor, mock_cov_cursor]

    sar_time = datetime(2026, 3, 12, 6, 0, 0, tzinfo=timezone.utc)
    res = await engine.correlate_detection(
        session=session,
        lat=50.5,
        lon=-1.5,
        acquisition_time=sar_time,
    )

    assert res["match_status"] == MatchStatus.DARK_VESSEL_CANDIDATE
    assert "Potential dark-vessel candidate" in res["provenance"]["explanation"]
    assert res["provenance"]["regional_reporting_vessels"] == 5


@pytest.mark.asyncio
async def test_insufficient_coverage_status_when_sparse():
    """Verify an unassociated target produces AIS-UNMATCHED-INSUFFICIENT-COVERAGE when regional traffic is sparse."""
    engine = AISSARCorrelationEngine(time_window_minutes=30)
    session = AsyncMock()

    # Nearby: empty
    mock_cand_cursor = MagicMock()
    mock_cand_cursor.fetchall.return_value = []
    # Regional: only 1 vessel
    mock_cov_cursor = MagicMock()
    mock_cov_cursor.fetchone.return_value = (1,)

    session.execute.side_effect = [mock_cand_cursor, mock_cov_cursor]

    sar_time = datetime(2026, 3, 12, 6, 0, 0, tzinfo=timezone.utc)
    res = await engine.correlate_detection(
        session=session,
        lat=50.5,
        lon=-1.5,
        acquisition_time=sar_time,
    )

    assert res["match_status"] == MatchStatus.UNMATCHED_INSUFFICIENT_COVERAGE
    assert "insufficient evidence" in res["provenance"]["explanation"]


@pytest.mark.asyncio
async def test_comparison_unavailable_when_no_coverage():
    """Verify an unassociated target produces AIS-COMPARISON-UNAVAILABLE when 0 vessels exist."""
    engine = AISSARCorrelationEngine(time_window_minutes=30)
    session = AsyncMock()

    mock_cand_cursor = MagicMock()
    mock_cand_cursor.fetchall.return_value = []
    mock_cov_cursor = MagicMock()
    mock_cov_cursor.fetchone.return_value = (0,)

    session.execute.side_effect = [mock_cand_cursor, mock_cov_cursor]

    sar_time = datetime(2026, 3, 12, 6, 0, 0, tzinfo=timezone.utc)
    res = await engine.correlate_detection(
        session=session,
        lat=50.5,
        lon=-1.5,
        acquisition_time=sar_time,
    )

    assert res["match_status"] == MatchStatus.COMPARISON_UNAVAILABLE
    assert "Historical AIS observation data unavailable" in res["provenance"]["explanation"]
