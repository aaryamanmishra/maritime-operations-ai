from datetime import datetime, timedelta
from typing import Any

from app.core.logging import logger
from app.domain.ml_performance.inference import TFTPerformancePredictor
from app.domain.weather_routing.models import (
    GeoCoordinate,
    NavigationalRoute,
    RouteOptimizationRequest,
    RouteOptimizationResponse,
    RouteSegment,
    RouteSummary,
)
from app.domain.weather_routing.physics import calculate_segment_physics
from app.infrastructure.adapters.open_meteo.adapter import OpenMeteoMarineAdapter
from app.routing.searoute.adapter import SeaRouteAdapter


class WeatherAwareRouteOptimizer:
    """
    Two-tier Weather-Aware Maritime Route Optimization Engine.
    
    1. Base Navigation: SeaRoute graph avoiding land.
    2. Environmental Layer: Open-Meteo Marine (waves, currents, wind, SST) via Redis cache.
    3. Performance ML Layer: Temporal Fusion Transformer predicting operational speed loss.
    4. Energy/Economics Layer: Naval architecture resistance, engine load, SFOC curve.
    5. Multi-Objective Cost Optimization: Evaluates base corridor vs weather-avoidance detours.
    """
    def __init__(
        self,
        searoute_adapter: SeaRouteAdapter = None,
        weather_adapter: OpenMeteoMarineAdapter = None,
        performance_predictor: TFTPerformancePredictor = None,
    ):
        self.searoute = searoute_adapter or SeaRouteAdapter()
        self.weather = weather_adapter or OpenMeteoMarineAdapter()
        self.predictor = performance_predictor or TFTPerformancePredictor.get_instance()

    async def optimize_route(self, request: RouteOptimizationRequest) -> RouteOptimizationResponse:
        logger.info(
            f"Optimizing route from ({request.origin.latitude}, {request.origin.longitude}) "
            f"to ({request.destination.latitude}, {request.destination.longitude}) "
            f"for vessel={request.vessel.vessel_type} at design_speed={request.vessel.design_speed_kn} kn"
        )

        # Step 1: Calculate Base Maritime Route via SeaRoute
        base_route_raw = self.searoute.calculate_base_maritime_route(request.origin, request.destination)
        
        # Step 2: Evaluate Base Route with Weather & Performance
        base_nav_route, base_cost = await self._evaluate_route_candidate(
            base_route_raw, request, route_type="base_maritime"
        )

        # Step 3: Explore Candidate Corridors / Lateral Detours
        candidate_detours = self.searoute.generate_alternative_corridors(base_route_raw, lateral_offset_nm=40.0)

        best_weather_route = base_nav_route
        best_cost = base_cost
        route_diverged = False

        for detour_raw in candidate_detours:
            try:
                candidate_route, candidate_cost = await self._evaluate_route_candidate(
                    detour_raw, request, route_type="weather_aware"
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_weather_route = candidate_route
                    route_diverged = True
                    logger.info(f"Weather-aware route diverged from base route! (Cost {candidate_cost:.1f} vs {base_cost:.1f})")
            except Exception as e:
                logger.warning(f"Error evaluating candidate corridor: {e}")

        # If base was better or same, the weather-aware route follows the base route
        if not route_diverged:
            # Create a weather_aware marked copy of base route
            best_weather_route = base_nav_route.model_copy(update={"route_type": "weather_aware"})

        # Summary comparison
        time_delta_hrs = best_weather_route.summary.estimated_duration_hours - base_nav_route.summary.estimated_duration_hours
        fuel_delta_t = best_weather_route.summary.total_estimated_fuel_tonnes - base_nav_route.summary.total_estimated_fuel_tonnes
        cost_delta_usd = best_weather_route.summary.total_estimated_cost_usd - base_nav_route.summary.total_estimated_cost_usd

        summary_comparison = {
            "distance_difference_nm": round(best_weather_route.summary.total_distance_nm - base_nav_route.summary.total_distance_nm, 2),
            "duration_difference_hours": round(time_delta_hrs, 2),
            "estimated_fuel_difference_tonnes": round(fuel_delta_t, 3),
            "estimated_cost_difference_usd": round(cost_delta_usd, 2),
            "base_max_wave_height_m": base_nav_route.summary.max_wave_height_m,
            "optimized_max_wave_height_m": best_weather_route.summary.max_wave_height_m,
            "wave_exposure_reduction_m": round(base_nav_route.summary.max_wave_height_m - best_weather_route.summary.max_wave_height_m, 2),
        }

        return RouteOptimizationResponse(
            request=request,
            base_route=base_nav_route,
            weather_aware_route=best_weather_route,
            route_diverged=route_diverged,
            summary_comparison=summary_comparison,
        )

    async def _evaluate_route_candidate(
        self,
        raw_route: dict[str, Any],
        request: RouteOptimizationRequest,
        route_type: str,
    ) -> tuple[NavigationalRoute, float]:
        """Fetch weather along segments, run TFT performance model, compute physics & cost."""
        segments_raw = raw_route["segments"]
        departure_time = request.departure_time
        calm_speed = request.vessel.design_speed_kn

        # Pre-estimate timestamps for weather query points
        points_to_query: list[tuple[float, float, datetime]] = []
        cum_time_hours = 0.0

        for seg in segments_raw:
            start_lat, start_lon = seg["start"]
            seg_time_at_calm = seg["distance_nm"] / max(calm_speed, 1.0)
            est_dt = departure_time + timedelta(hours=cum_time_hours)
            points_to_query.append((start_lat, start_lon, est_dt))
            cum_time_hours += seg_time_at_calm

        # Retrieve Marine Weather (cached in Redis / fetched from Open-Meteo)
        weather_map = await self.weather.get_weather_for_route_points(points_to_query)

        evaluated_segments: list[RouteSegment] = []
        current_time = departure_time
        total_distance_nm = 0.0
        total_energy_kwh = 0.0
        total_fuel_tonnes = 0.0
        total_cost_usd = 0.0
        total_speed_loss = 0.0
        wave_heights = []
        wind_speeds = []

        for idx, seg in enumerate(segments_raw):
            dist_nm = seg["distance_nm"]
            bearing = seg["bearing_deg"]
            start_lat, start_lon = seg["start"]
            end_lat, end_lon = seg["end"]

            # Match weather; must fail honestly if unavailable
            weather = weather_map.get(idx)
            if not weather:
                raise RuntimeError(
                    f"Required marine weather forecast data is unavailable for route waypoint {idx} "
                    f"at ({start_lat:.3f}, {start_lon:.3f}). Weather-aware calculation cannot proceed."
                )

            wave_heights.append(weather.wave_height_m)
            wind_speeds.append(weather.wind_speed_mps)

            # 1. Run TFT Deep-Learning Performance Prediction
            prediction = self.predictor.predict_performance(request.vessel, weather, bearing, dist_nm)
            pred_speed = prediction.predicted_sog_kn
            speed_loss = prediction.speed_loss_kn
            total_speed_loss += speed_loss

            # 2. Duration of segment
            duration_hrs = dist_nm / max(pred_speed, 1.5)
            seg_arrival = current_time + timedelta(hours=duration_hrs)

            # 3. Transparent Physics & Energy Model
            physics = calculate_segment_physics(
                request.vessel,
                request.fuel,
                weather,
                bearing,
                pred_speed,
                duration_hrs,
            )

            # Construct RouteSegment
            route_seg = RouteSegment(
                segment_index=idx,
                start_point=GeoCoordinate(latitude=start_lat, longitude=start_lon),
                end_point=GeoCoordinate(latitude=end_lat, longitude=end_lon),
                distance_nm=dist_nm,
                bearing_deg=bearing,
                estimated_departure=current_time,
                estimated_arrival=seg_arrival,
                duration_hours=round(duration_hrs, 2),
                weather=weather,
                calm_reference_speed_kn=round(calm_speed, 2),
                predicted_speed_kn=round(pred_speed, 2),
                speed_loss_kn=round(speed_loss, 2),
                current_speed_contribution_kn=physics["current_speed_contribution_kn"],
                total_resistance_kn=physics["r_total_kn"],
                estimated_brake_power_kw=physics["brake_power_kw"],
                estimated_energy_kwh=physics["energy_kwh"],
                estimated_fuel_tonnes=physics["fuel_tonnes"],
                estimated_fuel_cost_usd=physics["fuel_cost_usd"],
            )

            evaluated_segments.append(route_seg)
            total_distance_nm += dist_nm
            total_energy_kwh += physics["energy_kwh"]
            total_fuel_tonnes += physics["fuel_tonnes"]
            total_cost_usd += physics["fuel_cost_usd"]
            current_time = seg_arrival

        total_duration_hours = (current_time - departure_time).total_seconds() / 3600.0
        avg_speed_kn = total_distance_nm / max(total_duration_hours, 0.1)
        avg_speed_loss = total_speed_loss / max(len(evaluated_segments), 1)

        max_wave = max(wave_heights) if wave_heights else 0.0
        max_wind = max(wind_speeds) if wind_speeds else 0.0
        avg_wave = sum(wave_heights) / len(wave_heights) if wave_heights else 0.0

        exposure = "calm"
        if max_wave > 4.0 or max_wind > 17.0:
            exposure = "severe"
        elif max_wave > 2.5 or max_wind > 12.0:
            exposure = "rough"
        elif max_wave > 1.5 or max_wind > 8.0:
            exposure = "moderate"

        summary = RouteSummary(
            total_distance_nm=round(total_distance_nm, 2),
            estimated_duration_hours=round(total_duration_hours, 2),
            departure_time=departure_time,
            estimated_eta=current_time,
            mean_speed_kn=round(avg_speed_kn, 2),
            mean_speed_loss_kn=round(avg_speed_loss, 2),
            total_estimated_energy_kwh=round(total_energy_kwh, 1),
            total_estimated_fuel_tonnes=round(total_fuel_tonnes, 3),
            total_estimated_cost_usd=round(total_cost_usd, 2),
            max_wave_height_m=round(max_wave, 2),
            max_wind_speed_mps=round(max_wind, 2),
            avg_wave_height_m=round(avg_wave, 2),
            weather_exposure_level=exposure,
        )

        # 4. Multi-Objective Cost Calculation
        # Balances travel time, fuel, and weather penalty based on user priority
        prio = request.optimization_priority
        w_time = 1.0
        w_fuel = 1.0
        w_weather = 1.0

        if prio == "fuel_min":
            w_time = 0.5
            w_fuel = 2.0
            w_weather = 1.3
        elif prio == "time_min":
            w_time = 2.0
            w_fuel = 0.5
            w_weather = 0.8

        # Normalized cost metrics
        time_cost = total_duration_hours * 50.0 * w_time
        fuel_cost = total_cost_usd * 0.1 * w_fuel
        # Weather penalty: quadratic above 2.5m waves to heavily penalize storms
        weather_penalty = sum(max(h - 2.0, 0.0) ** 2 * 20.0 for h in wave_heights) * w_weather
        
        total_objective_cost = time_cost + fuel_cost + weather_penalty

        nav_route = NavigationalRoute(
            route_type=route_type,
            geometry=raw_route["geometry"],
            summary=summary,
            segments=evaluated_segments,
        )

        return nav_route, total_objective_cost
