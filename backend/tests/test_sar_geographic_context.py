import pytest
from datetime import datetime, UTC
from pathlib import Path

from app.domain.sar_intelligence.geographic_context import (
    GeographicContextStatus,
    GeographicContextValidator,
    GeographicContextResult,
)
from app.domain.sar_intelligence.models import (
    MatchStatus,
    SARDetectionItem,
)


@pytest.fixture
def validator():
    return GeographicContextValidator.get_instance()


def test_lop_nur_hard_negative_terrestrial_clutter(validator):
    """
    Hard-negative case from live CDSE burst 258592:
    Candidate at lat 39.42758, lon 90.09789 (Lop Nur, Xinjiang, China)
    Must be identified as terrestrial clutter, not an operational marine vessel.
    """
    lat, lon = 39.42758, 90.09789
    res = validator.validate_candidate(lat=lat, lon=lon)

    assert res.status == GeographicContextStatus.TERRESTRIAL_CLUTTER
    assert res.is_operational_vessel is False
    assert res.is_on_land is True
    assert res.distance_to_coast_km is not None
    assert res.distance_to_coast_km > 1000.0  # Actual: ~1848 km inland
    assert "inland from coastline" in res.reason
    assert "terrestrial radar clutter" in res.reason

    d = res.to_dict()
    assert d["status"] == "TERRESTRIAL_CLUTTER"
    assert d["is_operational_vessel"] is False
    assert d["distance_to_coast_km"] > 1000.0


def test_genuine_vessel_open_water(validator):
    """
    Genuine vessel over water (English Channel shipping lane: lat 50.5, lon 0.5).
    Must be confirmed as PLAUSIBLE_WATER and marked operational.
    """
    lat, lon = 50.5, 0.5
    res = validator.validate_candidate(lat=lat, lon=lon)

    assert res.status == GeographicContextStatus.PLAUSIBLE_WATER
    assert res.is_operational_vessel is True
    assert res.is_on_land is False
    assert res.distance_to_coast_km is not None
    assert res.distance_to_coast_km > 10.0  # >10 km offshore
    assert "offshore in open navigable waters" in res.reason


def test_ambiguous_shoreline_coastal_target(validator):
    """
    Target located in close proximity to the coastline / port boundary (Dover coast: lat 51.12, lon 1.32).
    Must be tagged as AMBIGUOUS_COASTAL, retain operational status, and document uncertainty.
    """
    lat, lon = 51.12, 1.32
    res = validator.validate_candidate(lat=lat, lon=lon)

    assert res.status == GeographicContextStatus.AMBIGUOUS_COASTAL
    assert res.is_operational_vessel is True
    assert res.distance_to_coast_km is not None
    assert res.distance_to_coast_km <= 10.0  # Inside coastal buffer
    assert "coastal ambiguity buffer" in res.reason


def test_unavailable_water_mask_data_fallback():
    """
    When water-mask dataset is missing, corrupted, or unavailable:
    Must fail gracefully to UNKNOWN status, document the limitation,
    and retain the candidate as operational (never drop valid vessels due to missing mask data).
    """
    empty_validator = GeographicContextValidator(data_path=Path("/nonexistent/mask.json"))
    res = empty_validator.validate_candidate(lat=39.42758, lon=90.09789)

    assert res.status == GeographicContextStatus.UNKNOWN
    assert res.is_operational_vessel is True
    assert "unavailable" in res.reason.lower()
    assert res.distance_to_coast_km is None


def test_dark_vessel_without_ais_over_water_preserved(validator):
    """
    Verify dark vessel preservation:
    A detection without AIS match (AIS-UNMATCHED) located over water must remain operational.
    Lack of AIS is the primary dark-vessel operational scenario, not a reason for rejection.
    """
    lat, lon = 20.0, 65.0  # Arabian Sea
    geo_res = validator.validate_candidate(lat=lat, lon=lon)

    assert geo_res.status == GeographicContextStatus.PLAUSIBLE_WATER
    assert geo_res.is_operational_vessel is True

    det_item = SARDetectionItem(
        detection_id="sar-det-dark-vessel-1",
        scene_id="S1A_IW_SLC_TEST_OCEAN",
        acquisition_time=datetime.now(UTC),
        latitude=lat,
        longitude=lon,
        confidence=0.88,
        model_version="yolo26s-opensar-slc-v1",
        modality="SLC",
        match_status=MatchStatus.UNMATCHED,
        surface_context=geo_res.status,
        is_operational_vessel=geo_res.is_operational_vessel,
        provenance={"geographic_context": geo_res.to_dict()},
    )

    assert det_item.match_status == MatchStatus.UNMATCHED
    assert det_item.is_operational_vessel is True
    assert det_item.surface_context == GeographicContextStatus.PLAUSIBLE_WATER


def test_out_of_bounds_coordinates(validator):
    """Test coordinates outside valid WGS84 range."""
    res = validator.validate_candidate(lat=95.0, lon=200.0)
    assert res.status == GeographicContextStatus.UNKNOWN
    assert "out of bounds" in res.reason
    assert res.is_operational_vessel is True
