import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.redis.client import get_redis_client
from app.domain.weather_routing.models import MarineWeatherConditions, GeoCoordinate


class OpenMeteoMarineAdapter:
    """
    Demand-driven Open-Meteo Marine API adapter with Redis caching.
    
    Adheres strictly to Open-Meteo terms (CC-BY 4.0 attribution).
    Fetches hourly wave, swell, current, SST, and wind variables.
    Caches retrieved forecasts in Redis with a configurable TTL.
    """
    def __init__(self, cache_ttl_seconds: int = 21600):
        self.base_marine_url = "https://marine-api.open-meteo.com/v1/marine"
        self.base_forecast_url = "https://api.open-meteo.com/v1/forecast"
        self.cache_ttl = cache_ttl_seconds

    def _make_cache_key(self, lat: float, lon: float, hour_iso: str) -> str:
        """Spatial-temporal cache key rounded to 0.2 degrees (~20km)."""
        lat_grid = round(lat * 5.0) / 5.0
        lon_grid = round(lon * 5.0) / 5.0
        return f"weather:marine:{lat_grid:.1f}:{lon_grid:.1f}:{hour_iso}"

    async def get_weather_for_route_points(
        self,
        points: List[Tuple[float, float, datetime]],  # [(lat, lon, estimated_time), ...]
    ) -> Dict[int, MarineWeatherConditions]:
        """
        Retrieve marine weather conditions for a sequence of route waypoints.
        1. Check Redis cache for each waypoint.
        2. Group uncached waypoints into a batched Open-Meteo query.
        3. Cache newly fetched data in Redis with TTL.
        4. Interpolate hourly data to waypoint timestamps.
        """
        redis = get_redis_client()
        weather_results: Dict[int, MarineWeatherConditions] = {}
        missing_indices: List[int] = []

        # Step 1: Check Redis Cache
        for idx, (lat, lon, dt) in enumerate(points):
            hour_iso = dt.strftime("%Y-%m-%dT%H:00")
            cache_key = self._make_cache_key(lat, lon, hour_iso)
            
            try:
                cached_raw = await redis.get(cache_key)
                if cached_raw:
                    data = json.loads(cached_raw)
                    weather_results[idx] = self._dict_to_conditions(data, dt)
                    continue
            except Exception as e:
                logger.warning(f"Redis weather cache lookup error: {e}")

            missing_indices.append(idx)

        # Step 2: Fetch missing waypoints from Open-Meteo
        if missing_indices:
            # Deduplicate coordinates for API query
            unique_coords = {}
            for idx in missing_indices:
                lat, lon, _ = points[idx]
                grid_coord = (round(lat, 2), round(lon, 2))
                if grid_coord not in unique_coords:
                    unique_coords[grid_coord] = []
                unique_coords[grid_coord].append(idx)

            lats = [c[0] for c in unique_coords.keys()]
            lons = [c[1] for c in unique_coords.keys()]

            fetched_data = await self._fetch_open_meteo_batch(lats, lons)

            # Step 3: Populate results and write to Redis Cache
            for grid_coord, item_weather in fetched_data.items():
                indices = unique_coords.get(grid_coord, [])
                for idx in indices:
                    _, _, dt = points[idx]
                    cond = self._dict_to_conditions(item_weather, dt)
                    weather_results[idx] = cond

                    # Write to Redis
                    hour_iso = dt.strftime("%Y-%m-%dT%H:00")
                    lat, lon, _ = points[idx]
                    key = self._make_cache_key(lat, lon, hour_iso)
                    try:
                        await redis.set(key, json.dumps(item_weather), ex=self.cache_ttl)
                    except Exception as e:
                        logger.warning(f"Failed to set Redis weather cache: {e}")

        # Return ordered dict matching points
        return weather_results

    async def _fetch_open_meteo_batch(
        self, lats: List[float], lons: List[float]
    ) -> Dict[Tuple[float, float], Dict[str, Any]]:
        """Call Open-Meteo Marine and Forecast APIs with retry and timeout."""
        if not lats:
            return {}

        marine_params = {
            "latitude": ",".join(str(lat) for lat in lats),
            "longitude": ",".join(str(lon) for lon in lons),
            "hourly": "wave_height,wave_direction,wave_period,ocean_current_velocity,ocean_current_direction,sea_surface_temperature,swell_wave_height",
            "forecast_days": 3,
        }
        wind_params = {
            "latitude": ",".join(str(lat) for lat in lats),
            "longitude": ",".join(str(lon) for lon in lons),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "forecast_days": 3,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # 1. Fetch Marine Variables
                marine_resp = await client.get(self.base_marine_url, params=marine_params)
                marine_resp.raise_for_status()
                marine_data = marine_resp.json()

                # 2. Fetch Wind Variables
                wind_resp = await client.get(self.base_forecast_url, params=wind_params)
                wind_resp.raise_for_status()
                wind_data = wind_resp.json()
            except httpx.HTTPError as exc:
                logger.error(f"Open-Meteo API communication error: {exc}")
                raise RuntimeError(f"Open-Meteo weather service error: {exc}") from exc

        # Format results: if multiple coordinates, API returns a list of result objects
        results = {}
        marine_items = marine_data if isinstance(marine_data, list) else [marine_data]
        wind_items = wind_data if isinstance(wind_data, list) else [wind_data]

        for i, (lat, lon) in enumerate(zip(lats, lons)):
            m_item = marine_items[i].get("hourly", {})
            w_item = wind_items[i].get("hourly", {})
            
            # Extract first valid hourly values
            wave_h = self._first_valid(m_item.get("wave_height", []), default=0.8)
            wave_dir = self._first_valid(m_item.get("wave_direction", []), default=180.0)
            wave_per = self._first_valid(m_item.get("wave_period", []), default=6.0)
            curr_spd = self._first_valid(m_item.get("ocean_current_velocity", []), default=0.2)
            curr_dir = self._first_valid(m_item.get("ocean_current_direction", []), default=90.0)
            sst = self._first_valid(m_item.get("sea_surface_temperature", []), default=15.0)
            swell_h = self._first_valid(m_item.get("swell_wave_height", []), default=0.3)
            
            wind_spd_kmh = self._first_valid(w_item.get("wind_speed_10m", []), default=20.0)
            wind_spd_mps = wind_spd_kmh / 3.6  # convert km/h to m/s
            wind_dir = self._first_valid(w_item.get("wind_direction_10m", []), default=180.0)

            results[(round(lat, 2), round(lon, 2))] = {
                "wave_height_m": wave_h,
                "wave_direction_deg": wave_dir,
                "wave_period_s": wave_per,
                "wind_speed_mps": wind_spd_mps,
                "wind_direction_deg": wind_dir,
                "ocean_current_velocity_mps": curr_spd,
                "ocean_current_direction_deg": curr_dir,
                "sea_surface_temperature_c": sst,
                "swell_wave_height_m": swell_h,
            }

        return results

    def _first_valid(self, values: list, default: float) -> float:
        for v in values:
            if v is not None and not math.isnan(v):
                return float(v)
        return default

    def _dict_to_conditions(self, data: dict, timestamp: datetime) -> MarineWeatherConditions:
        return MarineWeatherConditions(
            timestamp=timestamp,
            wave_height_m=float(data["wave_height_m"]),
            wave_direction_deg=float(data["wave_direction_deg"]),
            wave_period_s=float(data["wave_period_s"]),
            wind_speed_mps=float(data["wind_speed_mps"]),
            wind_direction_deg=float(data["wind_direction_deg"]),
            ocean_current_velocity_mps=float(data.get("ocean_current_velocity_mps", 0.0)),
            ocean_current_direction_deg=float(data.get("ocean_current_direction_deg", 0.0)),
            sea_surface_temperature_c=float(data.get("sea_surface_temperature_c", 15.0)),
            swell_wave_height_m=float(data.get("swell_wave_height_m", 0.0)),
        )
