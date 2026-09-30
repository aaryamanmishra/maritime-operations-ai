import pytest
from datetime import datetime, timezone
from app.domain.weather_routing.models import (
    GeoCoordinate,
    VesselCharacteristics,
    FuelParameters,
    RouteOptimizationRequest,
    MarineWeatherConditions,
)
from app.domain.weather_routing.physics import (
    calculate_relative_angle,
    calculate_segment_physics,
    estimate_displacement,
)
from app.routing.searoute.adapter import SeaRouteAdapter, haversine_distance_nm, calculate_bearing_deg
from app.infrastructure.adapters.open_meteo.adapter import OpenMeteoMarineAdapter
from app.domain.ml_performance.inference import TFTPerformancePredictor
from app.routing.optimizer import WeatherAwareRouteOptimizer


def test_haversine_distance_and_bearing():
    # Brest (48.38, -4.5) to Dover (51.12, 1.3)
    dist = haversine_distance_nm(48.38, -4.5, 51.12, 1.3)
    assert 250.0 < dist < 320.0

    bearing = calculate_bearing_deg(48.38, -4.5, 51.12, 1.3)
    assert 40.0 < bearing < 65.0  # North-East quadrant


def test_relative_angle_calculations():
    # Ship bearing 90° (East)
    # Head encounter: Wind/waves from 90° (East) -> 0° relative
    assert calculate_relative_angle(90.0, 90.0) == 0.0

    # Following encounter: Wind/waves from 270° (West) -> 180° or -180°
    assert abs(calculate_relative_angle(90.0, 270.0)) == 180.0

    # Beam encounter: Wind/waves from 0° (North) -> -90°
    assert calculate_relative_angle(90.0, 0.0) == -90.0

    # Wraparound across 0°
    assert calculate_relative_angle(10.0, 350.0) == -20.0


def test_searoute_base_maritime_route():
    adapter = SeaRouteAdapter()
    origin = GeoCoordinate(latitude=48.38, longitude=-4.5)  # Brest, France
    dest = GeoCoordinate(latitude=51.12, longitude=1.3)     # Dover, UK

    res = adapter.calculate_base_maritime_route(origin, dest)
    assert res["type"] == "Feature"
    assert res["geometry"]["type"] == "LineString"
    assert len(res["geometry"]["coordinates"]) >= 5
    assert res["properties"]["total_distance_nm"] > 250.0
    assert len(res["segments"]) >= 4

    # Verify each segment has valid kinematics
    for seg in res["segments"]:
        assert seg["distance_nm"] > 0.0
        assert 0.0 <= seg["bearing_deg"] < 360.0


def test_physics_resistance_and_fuel_estimation():
    vessel = VesselCharacteristics(
        vessel_type="cargo",
        length_m=180.0,
        beam_m=28.0,
        draft_m=10.0,
        design_speed_kn=14.0,
    )
    fuel = FuelParameters(fuel_price_per_tonne=600.0, fuel_type="VLSFO")
    
    # Calm weather
    calm_weather = MarineWeatherConditions(
        timestamp=datetime.now(timezone.utc),
        wave_height_m=0.5,
        wave_direction_deg=180.0,
        wave_period_s=4.0,
        wind_speed_mps=3.0,
        wind_direction_deg=180.0,
    )
    calm_res = calculate_segment_physics(vessel, fuel, calm_weather, bearing_deg=0.0, predicted_speed_kn=11.0, duration_hours=2.0)
    
    # Rough head sea weather
    rough_weather = MarineWeatherConditions(
        timestamp=datetime.now(timezone.utc),
        wave_height_m=3.5,
        wave_direction_deg=0.0,  # Directly opposing
        wave_period_s=8.0,
        wind_speed_mps=14.0,    # Strong opposing wind
        wind_direction_deg=0.0,
    )
    rough_res = calculate_segment_physics(vessel, fuel, rough_weather, bearing_deg=0.0, predicted_speed_kn=11.0, duration_hours=2.0)

    # In rough seas at same speed, wave, wind, total resistance, and fuel must be higher
    assert rough_res["r_total_kn"] > calm_res["r_total_kn"]
    assert rough_res["r_wave_kn"] > calm_res["r_wave_kn"]
    assert rough_res["r_wind_kn"] > calm_res["r_wind_kn"]
    assert rough_res["fuel_tonnes"] > calm_res["fuel_tonnes"]
    assert rough_res["fuel_cost_usd"] > calm_res["fuel_cost_usd"]


def test_weather_cache_key_generation():
    adapter = OpenMeteoMarineAdapter()
    key = adapter._make_cache_key(50.123, -1.876, "2026-09-30T14:00")
    # 50.123 rounds to 50.2, -1.876 rounds to -1.8 (0.2 degree grid)
    assert key == "weather:marine:50.2:-1.8:2026-09-30T14:00"


def test_tft_performance_predictor_speed_degradation():
    predictor = TFTPerformancePredictor.get_instance()
    vessel = VesselCharacteristics(
        vessel_type="tanker",
        length_m=240.0,
        beam_m=42.0,
        draft_m=14.0,
        design_speed_kn=13.5,
    )

    # 1. Calm weather
    calm = MarineWeatherConditions(
        timestamp=datetime.now(timezone.utc),
        wave_height_m=0.8,
        wave_direction_deg=180.0,
        wave_period_s=5.0,
        wind_speed_mps=4.0,
        wind_direction_deg=180.0,
    )
    pred_calm = predictor.predict_performance(vessel, calm, bearing_deg=0.0, distance_nm=30.0)

    # 2. Severe storm
    storm = MarineWeatherConditions(
        timestamp=datetime.now(timezone.utc),
        wave_height_m=5.5,
        wave_direction_deg=0.0,  # Head sea
        wave_period_s=11.0,
        wind_speed_mps=22.0,     # Near-gale
        wind_direction_deg=0.0,
    )
    pred_storm = predictor.predict_performance(vessel, storm, bearing_deg=0.0, distance_nm=30.0)

    assert pred_storm.predicted_sog_kn < pred_calm.predicted_sog_kn
    assert pred_storm.speed_loss_kn > pred_calm.speed_loss_kn
    assert pred_storm.speed_retention_ratio < 1.0


@pytest.mark.asyncio
async def test_weather_aware_route_optimizer_end_to_end():
    optimizer = WeatherAwareRouteOptimizer()
    request = RouteOptimizationRequest(
        origin=GeoCoordinate(latitude=49.0, longitude=-3.0),
        destination=GeoCoordinate(latitude=51.0, longitude=1.5),
        departure_time=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
        vessel=VesselCharacteristics(
            vessel_type="cargo",
            length_m=190.0,
            beam_m=28.0,
            draft_m=10.5,
            design_speed_kn=14.5,
        ),
        fuel=FuelParameters(fuel_price_per_tonne=620.0, fuel_type="VLSFO"),
        optimization_priority="balanced",
    )

    resp = await optimizer.optimize_route(request)
    assert resp.base_route is not None
    assert resp.weather_aware_route is not None
    assert resp.base_route.summary.total_distance_nm > 150.0
    assert resp.base_route.summary.total_estimated_fuel_tonnes > 0.0
    assert resp.base_route.summary.total_estimated_cost_usd > 0.0
    assert len(resp.base_route.segments) >= 2
    assert "ESTIMATES" in resp.disclaimer
