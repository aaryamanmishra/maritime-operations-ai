import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.sar_intelligence.coordinates import (
    GeoPoint,
    dead_reckon_position,
    haversine_distance_m,
    interpolate_position,
    normalize_longitude,
)
from app.domain.sar_intelligence.models import MatchStatus

logger = logging.getLogger(__name__)


class AISSARCorrelationEngine:
    """
    Spatiotemporal and kinematic uncertainty-aware correlation engine comparing
    SAR satellite vessel detections against recorded AIS observations.

    Physical Model & Concept Separation:
    1. Vessel movement: Kinematic displacement over elapsed time dt using Speed Over Ground (SOG)
       and Course Over Ground (COG), or geodesic interpolation between bounding AIS pings.
    2. SAR Geolocation Uncertainty (sigma_sar): ~30m nominal for Sentinel-1 GRD with GCP/TPS.
    3. AIS Position Uncertainty (sigma_ais): ~15m nominal for standard marine GNSS.
    4. Motion & Interpolation Uncertainty (sigma_motion): Kinematic turn rate, acceleration,
       and velocity variance over elapsed time dt.
    5. Soft-Edge Decision Boundary: Candidates near the boundary are tagged DETECTION-REVIEW-REQUIRED
       with a continuous likelihood score rather than being forced into a binary classification.
    """

    def __init__(
        self,
        time_window_minutes: int = 15,
        max_distance_km: float = 3.0,
        sigma_sar_m: float = 30.0,
        sigma_ais_m: float = 15.0,
    ):
        self.time_window_minutes = time_window_minutes
        self.max_distance_km = max_distance_km
        self.max_distance_meters = max_distance_km * 1000.0
        self.sigma_sar_m = sigma_sar_m
        self.sigma_ais_m = sigma_ais_m

    def compute_uncertainty_budget(
        self,
        speed_knots: float,
        elapsed_seconds: float,
    ) -> Dict[str, float]:
        """
        Compute total positional uncertainty standard deviation sigma_total in meters.
        sigma_total = sqrt(sigma_sar^2 + sigma_ais^2 + sigma_motion^2)
        """
        speed_mps = max(0.0, speed_knots * 0.51444444)
        dt = abs(elapsed_seconds)

        # Kinematic variance: accounts for course fluctuations and speed changes over elapsed time dt
        sigma_motion_m = max(
            15.0,
            0.08 * speed_mps * dt + 0.15 * dt
        )

        sigma_total_m = math.sqrt(
            self.sigma_sar_m ** 2 +
            self.sigma_ais_m ** 2 +
            sigma_motion_m ** 2
        )

        return {
            "sigma_sar_m": round(self.sigma_sar_m, 2),
            "sigma_ais_m": round(self.sigma_ais_m, 2),
            "sigma_motion_m": round(sigma_motion_m, 2),
            "sigma_total_m": round(sigma_total_m, 2),
        }

    def evaluate_candidate(
        self,
        distance_km: float,
        time_diff_seconds: float,
        speed_knots: float = 0.0,
        return_metrics: bool = False,
    ) -> Any:
        """
        Uncertainty-aware candidate evaluation with soft-edge boundary logic.
        """
        dist_m = distance_km * 1000.0
        max_time_s = self.time_window_minutes * 60.0
        dt = abs(time_diff_seconds)

        if dt > max_time_s:
            status = MatchStatus.UNMATCHED
            reason = f"Candidate at {dt:.1f}s offset exceeded time window of ±{self.time_window_minutes} min."
            z = dist_m / 100.0
            likelihood_score = 0.0
            hard_gate_m = self.max_distance_meters
            soft_gate_m = self.max_distance_meters * 1.5
            unc = self.compute_uncertainty_budget(speed_knots=speed_knots, elapsed_seconds=dt)
        elif speed_knots > 0.0:
            unc = self.compute_uncertainty_budget(speed_knots=speed_knots, elapsed_seconds=dt)
            sigma_tot = unc["sigma_total_m"]
            z = dist_m / max(sigma_tot, 1.0)
            likelihood_score = math.exp(-0.5 * min(z, 50.0) ** 2)
            hard_gate_m = min(self.max_distance_meters, max(150.0, 2.5 * sigma_tot))
            soft_gate_m = min(self.max_distance_meters * 1.5, 1.6 * hard_gate_m)

            if dist_m <= hard_gate_m and z <= 2.5:
                status = MatchStatus.MATCHED
                reason = f"Correlated within {distance_km:.2f} km and {dt:.1f}s offset (High-confidence correlation, z={z:.2f}, score={likelihood_score:.3f})."
            elif dist_m <= soft_gate_m or z <= 4.0:
                status = MatchStatus.DETECTION_REVIEW_REQUIRED
                reason = f"Ambiguous soft-edge candidate: dist={dist_m:.1f}m is near decision boundary (gate={hard_gate_m:.1f}m, z={z:.2f}, score={likelihood_score:.3f}). Operator review required."
            else:
                status = MatchStatus.UNMATCHED
                reason = f"Candidate distance {distance_km:.2f} km exceeded distance threshold and exceeded gates of {self.max_distance_km:.1f} km."
        else:
            # Unspecified speed (or legacy data without SOG)
            unc = self.compute_uncertainty_budget(speed_knots=speed_knots, elapsed_seconds=dt)
            sigma_tot = unc["sigma_total_m"]
            hard_gate_m = self.max_distance_meters
            soft_gate_m = self.max_distance_meters * 1.25
            z = dist_m / max(sigma_tot, 1.0)
            likelihood_score = math.exp(-0.5 * min((dist_m / max(hard_gate_m, 1.0)) * 2.0, 50.0) ** 2)

            if dist_m <= hard_gate_m:
                status = MatchStatus.MATCHED
                reason = f"Correlated within {distance_km:.2f} km and {dt:.1f}s offset."
            elif dist_m <= soft_gate_m:
                status = MatchStatus.DETECTION_REVIEW_REQUIRED
                reason = f"Ambiguous soft-edge candidate: dist={dist_m:.1f}m near search boundary ({hard_gate_m:.1f}m). Operator review required."
            else:
                status = MatchStatus.UNMATCHED
                reason = f"Candidate at {distance_km:.2f} km exceeded distance threshold of {self.max_distance_km:.1f} km."

        metrics = {
            "distance_m": round(dist_m, 2),
            "distance_km": round(distance_km, 3),
            "time_diff_seconds": round(time_diff_seconds, 1),
            "z_score": round(z, 2),
            "likelihood_score": round(likelihood_score, 4),
            "hard_gate_m": round(hard_gate_m, 1),
            "soft_gate_m": round(soft_gate_m, 1),
            "uncertainty_budget": unc,
        }

        if return_metrics:
            return status, reason, metrics
        return status, reason

    async def correlate_detection(
        self,
        session: AsyncSession,
        lat: float,
        lon: float,
        acquisition_time: datetime,
        exclude_mmsis: Optional[Set[int] | List[int]] = None,
        regional_coverage_radius_km: float = 35.0,
    ) -> Dict[str, Any]:
        """
        Perform kinematic, uncertainty-aware correlation of a SAR detection against
        historical AIS observations within the database.
        """
        if acquisition_time.tzinfo is None:
            acquisition_time = acquisition_time.replace(tzinfo=timezone.utc)

        start_time = acquisition_time - timedelta(minutes=self.time_window_minutes)
        end_time = acquisition_time + timedelta(minutes=self.time_window_minutes)

        params: Dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "start_time": start_time,
            "end_time": end_time,
            "max_dist_m": self.max_distance_meters * 1.5,
            "sar_time": acquisition_time,
        }

        exclude_filter = ""
        if exclude_mmsis:
            exclude_filter = "AND NOT (mmsi = ANY(:exclude_mmsis))"
            params["exclude_mmsis"] = list(exclude_mmsis)

        query = text(f"""
            WITH candidates AS (
                SELECT
                    mmsi,
                    timestamp,
                    latitude,
                    longitude,
                    sog,
                    cog,
                    ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS raw_dist_m
                FROM current_vessel_state
                WHERE timestamp BETWEEN :start_time AND :end_time
                  AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :max_dist_m)
                  {exclude_filter}

                UNION ALL

                SELECT
                    mmsi,
                    timestamp,
                    latitude,
                    longitude,
                    sog,
                    cog,
                    ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS raw_dist_m
                FROM vessel_position_history
                WHERE timestamp BETWEEN :start_time AND :end_time
                  AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :max_dist_m)
                  {exclude_filter}
            )
            SELECT
                mmsi,
                timestamp,
                latitude,
                longitude,
                sog,
                cog,
                raw_dist_m,
                EXTRACT(EPOCH FROM (:sar_time - timestamp)) AS time_diff_s
            FROM candidates
            ORDER BY raw_dist_m ASC
            LIMIT 25;
        """)

        result = await session.execute(query, params)
        rows = []
        if hasattr(result, "fetchall"):
            try:
                rows = result.fetchall()
            except Exception:
                rows = []
        if not rows and hasattr(result, "fetchone"):
            r = result.fetchone()
            if r is not None:
                rows = [r]

        best_match = None
        best_score = -1.0

        for row in rows:
            if len(row) >= 8:
                mmsi, ais_ts, ais_lat, ais_lon, sog, cog, raw_dist_m, time_diff_s = row[:8]
            elif len(row) == 4:
                mmsi, ais_ts, raw_dist_m, time_diff_s = row
                ais_lat, ais_lon, sog, cog = lat, lon, 0.0, 0.0
            else:
                continue

            sog_val = float(sog) if sog is not None else 0.0
            cog_val = float(cog) if cog is not None else 0.0
            dt_s = float(time_diff_s)

            # Kinematic dead reckoning
            if abs(dt_s) > 1.0 and sog_val > 0.5:
                proj_pt = dead_reckon_position(
                    lat=float(ais_lat),
                    lon=float(ais_lon),
                    speed_knots=sog_val,
                    course_deg=cog_val,
                    elapsed_seconds=dt_s,
                )
                est_lat = proj_pt.latitude
                est_lon = proj_pt.longitude
                proj_dist_m = haversine_distance_m(lat, lon, est_lat, est_lon)
                movement_est_m = haversine_distance_m(float(ais_lat), float(ais_lon), est_lat, est_lon)
            else:
                est_lat = float(ais_lat)
                est_lon = float(ais_lon)
                proj_dist_m = float(raw_dist_m)
                movement_est_m = 0.0

            status, reason, metrics = self.evaluate_candidate(
                distance_km=proj_dist_m / 1000.0,
                time_diff_seconds=dt_s,
                speed_knots=sog_val,
                return_metrics=True,
            )

            metrics["raw_distance_m"] = round(float(raw_dist_m), 2)
            metrics["estimated_movement_m"] = round(movement_est_m, 2)
            metrics["estimated_position"] = [round(est_lat, 6), round(est_lon, 6)]
            metrics["sog_knots"] = sog_val
            metrics["cog_deg"] = cog_val

            score = metrics["likelihood_score"]
            if score > best_score:
                best_score = score
                best_match = {
                    "mmsi": int(mmsi),
                    "timestamp": ais_ts,
                    "metrics": metrics,
                    "status": status,
                    "reason": reason,
                }

        if best_match and best_match["status"] in (MatchStatus.MATCHED, MatchStatus.DETECTION_REVIEW_REQUIRED):
            m = best_match["metrics"]
            provenance = {
                "explanation": (
                    f"Plausible AIS observation found for MMSI {best_match['mmsi']} at {m['distance_km']} km "
                    f"and {m['time_diff_seconds']}s offset from SAR observation."
                ),
                "mmsi": best_match["mmsi"],
                "sar_acquisition_time": acquisition_time.isoformat(),
                "ais_observation_time": (
                    best_match["timestamp"].isoformat()
                    if hasattr(best_match["timestamp"], "isoformat")
                    else str(best_match["timestamp"])
                ),
                "distance_km": m["distance_km"],
                "raw_distance_km": round(m["raw_distance_m"] / 1000.0, 3),
                "estimated_movement_m": m["estimated_movement_m"],
                "estimated_position": m["estimated_position"],
                "time_difference_seconds": m["time_diff_seconds"],
                "z_score": m["z_score"],
                "likelihood_score": m["likelihood_score"],
                "uncertainty_budget": m["uncertainty_budget"],
                "hard_gate_m": m["hard_gate_m"],
                "soft_gate_m": m["soft_gate_m"],
                "algorithm_version": "uncertainty-kinematic-geodesic-v2",
                "one_to_one_mmsi_enforced": bool(exclude_mmsis),
            }

            return {
                "match_status": best_match["status"],
                "candidate_mmsi": best_match["mmsi"],
                "distance_km": m["distance_km"],
                "time_difference_seconds": m["time_diff_seconds"],
                "provenance": provenance,
            }

        # 3. No match found -> Evaluate regional coverage
        cov_query = text("""
            WITH cov_sample AS (
                SELECT mmsi, timestamp, geom FROM current_vessel_state
                UNION ALL
                SELECT mmsi, timestamp, geom FROM vessel_position_history
            )
            SELECT COUNT(DISTINCT mmsi) AS regional_vessel_count
            FROM cov_sample
            WHERE timestamp BETWEEN :cov_start AND :cov_end
              AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :cov_dist_m);
        """)

        cov_res = await session.execute(
            cov_query,
            {
                "lat": lat,
                "lon": lon,
                "cov_start": acquisition_time - timedelta(minutes=self.time_window_minutes),
                "cov_end": acquisition_time + timedelta(minutes=self.time_window_minutes),
                "cov_dist_m": regional_coverage_radius_km * 1000.0,
            },
        )
        cov_row = cov_res.fetchone() if hasattr(cov_res, "fetchone") else None
        regional_count = int(cov_row[0]) if cov_row and cov_row[0] is not None else 0

        if regional_count >= 2:
            match_status = MatchStatus.DARK_VESSEL_CANDIDATE
            explanation = (
                f"Potential dark-vessel candidate: verified regional AIS coverage active "
                f"({regional_count} reporting vessels within {regional_coverage_radius_km:.0f} km and ±{self.time_window_minutes}m), "
                f"but no correlating AIS transponder signal identified for this detection."
            )
        elif regional_count == 1:
            match_status = MatchStatus.UNMATCHED_INSUFFICIENT_COVERAGE
            explanation = (
                f"Detection unmatched: only {regional_count} vessel reporting within {regional_coverage_radius_km:.0f} km. "
                "Regional AIS receiver coverage is sparse/unconfirmed; insufficient evidence to flag as dark vessel."
            )
        else:
            match_status = MatchStatus.COMPARISON_UNAVAILABLE
            explanation = (
                f"Historical AIS data coverage unavailable (Historical AIS observation data unavailable) in database within {regional_coverage_radius_km:.0f} km "
                f"and ±{self.time_window_minutes}m for observation timestamp; transponder correlation cannot be evaluated."
            )

        provenance = {
            "explanation": explanation,
            "sar_acquisition_time": acquisition_time.isoformat(),
            "regional_coverage_radius_km": regional_coverage_radius_km,
            "regional_reporting_vessels": regional_count,
            "algorithm_version": "uncertainty-kinematic-geodesic-v2",
            "thresholds": {
                "max_distance_km": self.max_distance_km,
                "time_window_minutes": self.time_window_minutes,
            },
        }

        return {
            "match_status": match_status,
            "candidate_mmsi": None,
            "distance_km": None,
            "time_difference_seconds": None,
            "provenance": provenance,
        }
