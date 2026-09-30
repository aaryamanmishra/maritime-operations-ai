import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.sar_intelligence.models import MatchStatus

logger = logging.getLogger(__name__)


class AISSARCorrelationEngine:
    """
    Spatiotemporal correlation engine comparing SAR satellite vessel detections
    against recorded AIS observations within configured tolerances.
    
    Terminology:
    - AIS-MATCHED: A plausible AIS position report was identified within tolerance.
    - AIS-UNMATCHED: No AIS vessel report correlated within configured tolerances.
    """

    def __init__(
        self,
        time_window_minutes: int = 15,
        max_distance_km: float = 3.0,
    ):
        self.time_window_minutes = time_window_minutes
        self.max_distance_km = max_distance_km
        self.max_distance_meters = max_distance_km * 1000.0

    def evaluate_candidate(
        self,
        distance_km: float,
        time_diff_seconds: float,
    ) -> tuple[MatchStatus, str]:
        """
        Evaluate candidate proximity against configured spatial and temporal thresholds.
        """
        max_time_diff_s = self.time_window_minutes * 60.0
        if distance_km > self.max_distance_km:
            return (
                MatchStatus.UNMATCHED,
                f"Candidate at {distance_km:.2f} km exceeded distance threshold of {self.max_distance_km:.1f} km.",
            )
        if abs(time_diff_seconds) > max_time_diff_s:
            return (
                MatchStatus.UNMATCHED,
                f"Candidate at {abs(time_diff_seconds):.1f}s offset exceeded time window of ±{self.time_window_minutes} min.",
            )
        return (
            MatchStatus.MATCHED,
            f"Correlated within {distance_km:.2f} km and {abs(time_diff_seconds):.1f}s offset.",
        )


    async def correlate_detection(
        self,
        session: AsyncSession,
        lat: float,
        lon: float,
        acquisition_time: datetime,
    ) -> dict[str, Any]:
        """
        Find the nearest plausible AIS candidate within [acquisition_time - window, acquisition_time + window]
        and within max_distance_km of (lat, lon).
        """
        start_time = acquisition_time - timedelta(minutes=self.time_window_minutes)
        end_time = acquisition_time + timedelta(minutes=self.time_window_minutes)

        query = text("""
            WITH candidates AS (
                -- 1. Check current_vessel_state
                SELECT
                    mmsi,
                    timestamp,
                    ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS dist_m
                FROM current_vessel_state
                WHERE timestamp BETWEEN :start_time AND :end_time
                  AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :max_dist_m)
                
                UNION ALL
                
                -- 2. Check vessel_position_history
                SELECT
                    mmsi,
                    timestamp,
                    ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS dist_m
                FROM vessel_position_history
                WHERE timestamp BETWEEN :start_time AND :end_time
                  AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :max_dist_m)
            )
            SELECT
                mmsi,
                timestamp,
                dist_m,
                ABS(EXTRACT(EPOCH FROM (timestamp - :sar_time))) AS time_diff_s
            FROM candidates
            ORDER BY dist_m ASC, time_diff_s ASC
            LIMIT 1;
        """)

        result = await session.execute(
            query,
            {
                "lat": lat,
                "lon": lon,
                "start_time": start_time,
                "end_time": end_time,
                "max_dist_m": self.max_distance_meters,
                "sar_time": acquisition_time,
            },
        )
        row = result.fetchone()

        if row:
            mmsi, ais_timestamp, dist_m, time_diff_s = row
            dist_km = round(dist_m / 1000.0, 3)
            time_diff_s = round(float(time_diff_s), 1)

            provenance = {
                "explanation": (
                    f"Plausible AIS observation found for MMSI {mmsi} at {dist_km} km "
                    f"and {time_diff_s}s offset from SAR observation."
                ),
                "matching_thresholds": {
                    "max_distance_km": self.max_distance_km,
                    "max_time_window_minutes": self.time_window_minutes,
                },
                "sar_acquisition_time": acquisition_time.isoformat(),
                "ais_observation_time": ais_timestamp.isoformat() if hasattr(ais_timestamp, "isoformat") else str(ais_timestamp),
                "distance_km": dist_km,
                "time_difference_seconds": time_diff_s,
                "algorithm_version": "spatiotemporal-geodetic-v1",
            }

            return {
                "match_status": MatchStatus.MATCHED,
                "candidate_mmsi": int(mmsi),
                "distance_km": dist_km,
                "time_difference_seconds": time_diff_s,
                "provenance": provenance,
            }

        # No candidate correlated
        provenance = {
            "explanation": (
                f"No broadcasting AIS vessel correlated within {self.max_distance_km} km "
                f"and ±{self.time_window_minutes} minutes of SAR observation."
            ),
            "matching_thresholds": {
                "max_distance_km": self.max_distance_km,
                "max_time_window_minutes": self.time_window_minutes,
            },
            "sar_acquisition_time": acquisition_time.isoformat(),
            "algorithm_version": "spatiotemporal-geodetic-v1",
        }

        return {
            "match_status": MatchStatus.UNMATCHED,
            "candidate_mmsi": None,
            "distance_km": None,
            "time_difference_seconds": None,
            "provenance": provenance,
        }
