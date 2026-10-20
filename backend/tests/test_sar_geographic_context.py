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


def test_coastal_vessels_recovered_from_terrestrial_clutter(validator):
    """
    Verify recovery of key coastal vessels previously false-rejected by coarse 1:110m polygons:
    1. Lagos coastal roadstead (DB_OPENSAR_VD_100): 6.255°N, 4.515°E
    2. Port of Esbjerg tidal channel, Denmark (DB_OPENSAR_VD_3326): 55.449°N, 8.492°E
    3. Escravos River delta mouth, Nigeria (DB_OPENSAR_VD_3): 5.842°N, 5.173°E
    4. Wouri estuary, Cameroon (DB_OPENSAR_VD_5513): 4.235°N, 8.914°E

    Must be recognized as operational vessels within coastal context (not continental clutter).
    """
    cases = [
        ("Lagos Roadstead (VD_100)", 6.2546, 4.5148),
        ("Port of Esbjerg (VD_3326)", 55.4488, 8.4924),
        ("Escravos River Delta (VD_3)", 5.8423, 5.1733),
        ("Cameroon Estuary (VD_5513)", 4.2346, 8.9136),
    ]

    for label, lat, lon in cases:
        res = validator.validate_candidate(lat=lat, lon=lon)
        assert res.status != GeographicContextStatus.TERRESTRIAL_CLUTTER, (
            f"{label} at ({lat}, {lon}) should not be classified as TERRESTRIAL_CLUTTER, got {res.status}"
        )
        assert res.is_operational_vessel is True, f"{label} must remain operational vessel"
        assert res.distance_to_coast_km is not None
        assert res.distance_to_coast_km <= 10.0, f"{label} distance to coast should be <= 10.0 km"


def test_genuine_continental_clutter_strictly_rejected(validator):
    """
    Verify that genuine inland continental clutter targets remain strictly rejected:
    1. Lop Nur hard negative (Xinjiang, China): 39.42758°N, 90.09789°E (~1760 km inland)
    2. Mainland Nigeria inland clutter (VD_1523): 6.71213°N, 5.37473°E (~90 km inland)
    3. Mainland Nigeria inland clutter (VD_1578): 6.64336°N, 5.31465°E (~82 km inland)
    """
    inland_targets = [
        ("Lop Nur Playa", 39.42758, 90.09789, 1000.0),
        ("Mainland Nigeria (VD_1523)", 6.71213, 5.37473, 50.0),
        ("Mainland Nigeria (VD_1578)", 6.64336, 5.31465, 50.0),
    ]

    for label, lat, lon, min_dist in inland_targets:
        res = validator.validate_candidate(lat=lat, lon=lon)
        assert res.status == GeographicContextStatus.TERRESTRIAL_CLUTTER, (
            f"{label} must be classified as TERRESTRIAL_CLUTTER, got {res.status}"
        )
        assert res.is_operational_vessel is False
        assert res.is_on_land is True
        assert res.distance_to_coast_km > min_dist


def test_all_36_forensic_coastal_vessels_recovery(validator):
    """
    Regression test across all 36 coastal true-positive detections identified in the forensic audit.
    Zero of the 36 must be rejected as TERRESTRIAL_CLUTTER; 100% must be retained as operational vessels.
    """
    coords_36 = [
        (6.2546, 4.51484), (6.36771, 4.07805), (6.36823, 4.07636), (6.40869, 4.01245),
        (6.35419, 3.89827), (6.36334, 3.89653), (6.37378, 3.89049), (6.36023, 3.88985),
        (6.35326, 3.53333), (6.36536, 3.43931), (6.37387, 3.43845), (6.37685, 3.41553),
        (6.43537, 3.3952), (6.41811, 3.40342), (6.41499, 3.38289), (6.3743, 3.37782),
        (6.3645, 3.3647), (6.37404, 3.38135), (6.38081, 3.38252), (6.35841, 3.37681),
        (6.45593, 3.37571), (6.4314, 3.36259), (5.84234, 5.17326), (55.4488, 8.49243),
        (55.44074, 8.47491), (5.83279, 4.99529), (5.8297, 4.97354), (5.84548, 4.97278),
        (5.85791, 4.9529), (5.86105, 4.95125), (5.86538, 4.9373), (5.86973, 4.93238),
        (5.86515, 4.92696), (4.23455, 8.91358), (5.93921, 4.82645), (6.24767, 4.54819),
    ]

    terrestrial_clutter_count = 0
    operational_count = 0

    for lat, lon in coords_36:
        res = validator.validate_candidate(lat=lat, lon=lon)
        if res.status == GeographicContextStatus.TERRESTRIAL_CLUTTER:
            terrestrial_clutter_count += 1
        if res.is_operational_vessel:
            operational_count += 1

    assert terrestrial_clutter_count == 0, f"Expected 0 TERRESTRIAL_CLUTTER, found {terrestrial_clutter_count}"
    assert operational_count == len(coords_36), f"Expected all 36 to be operational, got {operational_count}"
