import pytest
from datetime import datetime, timezone
from app.domain.vessel_traffic.models import (
    BoundingBox, NormalizedVesselEvent, haversine_distance_m, heading_delta_deg, should_record_history
)
from app.infrastructure.adapters.aisstream.adapter import AISStreamAdapter


def test_haversine_distance():
    # Distance between Dover (51.1279, 1.3134) and Calais (50.9513, 1.8587) is approx 43.6 km (43,600 m)
    dist = haversine_distance_m(51.1279, 1.3134, 50.9513, 1.8587)
    assert 40000 < dist < 46000


def test_heading_delta_wraparound():
    # Normal delta
    assert heading_delta_deg(10.0, 25.0) == 15.0
    # Wraparound across 360° / 0°
    assert heading_delta_deg(355.0, 5.0) == 10.0
    assert heading_delta_deg(5.0, 355.0) == 10.0
    assert heading_delta_deg(180.0, 180.0) == 0.0
    assert heading_delta_deg(0.0, 180.0) == 180.0
    # None handling
    assert heading_delta_deg(None, 45.0) == 0.0


def test_should_record_history_criteria():
    t0 = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 30, 12, 0, 30, tzinfo=timezone.utc)  # 30s elapsed

    # 1. First point ever -> should record
    e1 = NormalizedVesselEvent(mmsi=123456789, latitude=50.0, longitude=1.0, timestamp=t0)
    assert should_record_history(None, None, None, None, e1) is True

    # 2. Insignificant movement (<100m, same heading, <180s) -> should NOT record
    # 0.0001 deg latitude is approx 11 meters
    e2 = NormalizedVesselEvent(mmsi=123456789, latitude=50.0001, longitude=1.0, heading=90.0, timestamp=t1)
    assert should_record_history(50.0, 1.0, 90.0, t0, e2, distance_threshold_m=100.0) is False

    # 3. Exceeded distance (> 100m) -> should record
    # 0.002 deg latitude is approx 222 meters
    e3 = NormalizedVesselEvent(mmsi=123456789, latitude=50.002, longitude=1.0, heading=90.0, timestamp=t1)
    assert should_record_history(50.0, 1.0, 90.0, t0, e3, distance_threshold_m=100.0) is True

    # 4. Heading turn (> 5 degrees) -> should record
    e4 = NormalizedVesselEvent(mmsi=123456789, latitude=50.0001, longitude=1.0, heading=98.0, timestamp=t1)
    assert should_record_history(50.0, 1.0, 90.0, t0, e4, heading_threshold_deg=5.0) is True

    # 5. Elapsed time (> 180s) -> should record
    t_later = datetime(2026, 9, 30, 12, 4, 0, tzinfo=timezone.utc)  # 240s elapsed
    e5 = NormalizedVesselEvent(mmsi=123456789, latitude=50.0001, longitude=1.0, heading=90.0, timestamp=t_later)
    assert should_record_history(50.0, 1.0, 90.0, t0, e5, time_threshold_s=180) is True


def test_bounding_box_validation():
    # Valid bounding box
    bbox = BoundingBox(min_lat=49.0, min_lon=-2.0, max_lat=51.0, max_lon=2.0)
    assert bbox.contains(50.0, 0.0) is True
    assert bbox.contains(55.0, 0.0) is False

    # Invalid latitude order
    with pytest.raises(ValueError):
        BoundingBox(min_lat=52.0, min_lon=-2.0, max_lat=50.0, max_lon=2.0)

    # Invalid longitude order
    with pytest.raises(ValueError):
        BoundingBox(min_lat=49.0, min_lon=3.0, max_lat=50.0, max_lon=1.0)


def test_aisstream_adapter_position_report():
    raw = {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": 246591000,
            "ShipName": "VANGUARD           ",
            "latitude": 50.888,
            "longitude": 1.130,
            "time_utc": "2026-09-30 08:00:00.123 +0000 UTC"
        },
        "Message": {
            "PositionReport": {
                "Sog": 14.2,
                "Cog": 225.0,
                "TrueHeading": 224,
                "NavigationalStatus": 0,
                "Latitude": 50.888,
                "Longitude": 1.130
            }
        }
    }

    event = AISStreamAdapter.normalize_message(raw)
    assert event is not None
    assert event.mmsi == 246591000
    assert event.ship_name == "VANGUARD"
    assert event.latitude == 50.888
    assert event.longitude == 1.130
    assert event.sog == 14.2
    assert event.heading == 224
    assert event.nav_status == 0


def test_aisstream_adapter_heading_511_normalization():
    raw = {
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": 123456789, "latitude": 50.0, "longitude": 1.0},
        "Message": {
            "PositionReport": {
                "Sog": 5.0,
                "Cog": 100.0,
                "TrueHeading": 511,  # 511 means heading unavailable
                "Latitude": 50.0,
                "Longitude": 1.0
            }
        }
    }
    event = AISStreamAdapter.normalize_message(raw)
    assert event is not None
    assert event.heading is None  # Normalized to None


def test_aisstream_adapter_ship_static_data():
    raw = {
        "MessageType": "ShipStaticData",
        "MetaData": {
            "MMSI": 566028000,
            "ShipName": "PACIFIC EXPLORER   ",
            "latitude": 50.5,
            "longitude": 0.5,
            "time_utc": "2026-09-30 08:00:00 +0000 UTC"
        },
        "Message": {
            "ShipStaticData": {
                "Name": "PACIFIC EXPLORER   ",
                "ImoNumber": 9438482,
                "CallSign": "9V9398 ",
                "Type": 70,
                "MaximumStaticDraught": 8.5,
                "Destination": "ROTTERDAM          ",
                "Dimension": {"A": 150, "B": 30, "C": 20, "D": 10}
            }
        }
    }

    event = AISStreamAdapter.normalize_message(raw)
    assert event is not None
    assert event.mmsi == 566028000
    assert event.ship_name == "PACIFIC EXPLORER"
    assert event.imo == 9438482
    assert event.callsign == "9V9398"
    assert event.ship_type == 70
    assert event.destination == "ROTTERDAM"
    assert event.length == 180.0  # 150 + 30
    assert event.beam == 30.0    # 20 + 10
    assert event.draught == 8.5
