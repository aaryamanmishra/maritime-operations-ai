import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from searoute.utils import haversine, pnpoly

logger = logging.getLogger(__name__)


class GeographicContextStatus(str, Enum):
    PLAUSIBLE_WATER = "PLAUSIBLE_WATER"
    TERRESTRIAL_CLUTTER = "TERRESTRIAL_CLUTTER"
    AMBIGUOUS_COASTAL = "AMBIGUOUS_COASTAL"
    UNKNOWN = "UNKNOWN"


class GeographicContextResult:
    def __init__(
        self,
        status: GeographicContextStatus,
        reason: str,
        is_operational_vessel: bool,
        distance_to_coast_km: float | None = None,
        is_on_land: bool | None = None,
        data_source: str = "Natural Earth 1:110m Physical Land",
    ):
        self.status = status
        self.reason = reason
        self.is_operational_vessel = is_operational_vessel
        self.distance_to_coast_km = distance_to_coast_km
        self.is_on_land = is_on_land
        self.data_source = data_source

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "is_operational_vessel": self.is_operational_vessel,
            "distance_to_coast_km": (
                round(self.distance_to_coast_km, 2)
                if self.distance_to_coast_km is not None
                else None
            ),
            "is_on_land": self.is_on_land,
            "data_source": self.data_source,
        }

    def __repr__(self) -> str:
        return f"<GeographicContextResult status={self.status.value} operational={self.is_operational_vessel} coast_dist={self.distance_to_coast_km}km>"


class PreparedPolygon:
    """Pre-computed polygon structure with bounding box pre-filter for microsecond pnpoly lookups."""

    def __init__(self, exterior: list[list[float]]):
        self.exterior = exterior
        self.vx = [p[0] for p in exterior]
        self.vy = [p[1] for p in exterior]
        self.min_x = min(self.vx)
        self.max_x = max(self.vx)
        self.min_y = min(self.vy)
        self.max_y = max(self.vy)
        self.nvert = len(self.vx)
        self.ext_arr = np.array(exterior, dtype=np.float64) if len(exterior) > 1 else None


class GeographicContextValidator:
    """
    Validates the surface environment and geographic context of SAR vessel candidates.
    Distinguishes genuine marine targets from inland terrestrial radar clutter
    (mountains, salt playas, rock outcrops) and flags coastal ambiguity.
    
    Guarantees:
    - Never rejects dark vessels merely for lacking AIS.
    - Transparent uncertainty reporting for nearshore/coastal boundaries and unavailable data.
    - Fully deterministic execution based on Natural Earth public domain geospatial boundaries.
    """

    _instance: "GeographicContextValidator | None" = None

    def __init__(
        self,
        data_path: str | Path | None = None,
        coastal_ambiguity_buffer_km: float = 10.0,
    ):
        self.coastal_ambiguity_buffer_km = coastal_ambiguity_buffer_km
        self.data_source_name = "Natural Earth Physical Land"
        self._polygons: list[PreparedPolygon] = []
        self._is_loaded: bool = False

        resolved_path = self._resolve_data_path(data_path)
        if resolved_path and resolved_path.exists():
            self._load_dataset(resolved_path)
        else:
            logger.warning(
                "Geographic context dataset not found at %s. Land/water mask validation will report UNKNOWN.",
                resolved_path,
            )

    @classmethod
    def get_instance(cls) -> "GeographicContextValidator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _resolve_data_path(data_path: str | Path | None) -> Path | None:
        if data_path:
            return Path(data_path)
        # Search candidate locations (preferring high-resolution 10m dataset)
        base_dir = Path(__file__).resolve().parent
        candidates = [
            base_dir / "data" / "ne_10m_land.json",
            Path("backend/app/domain/sar_intelligence/data/ne_10m_land.json"),
            Path("app/domain/sar_intelligence/data/ne_10m_land.json"),
            base_dir / "data" / "ne_110m_land.json",
            Path("backend/app/domain/sar_intelligence/data/ne_110m_land.json"),
            Path("app/domain/sar_intelligence/data/ne_110m_land.json"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _load_dataset(self, path: Path) -> None:
        try:
            if "10m" in str(path):
                self.data_source_name = "Natural Earth 1:10m Physical Land"
            elif "50m" in str(path):
                self.data_source_name = "Natural Earth 1:50m Physical Land"
            elif "110m" in str(path):
                self.data_source_name = "Natural Earth 1:110m Physical Land"
            else:
                self.data_source_name = f"Natural Earth Land ({path.name})"

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            features = data.get("features", [])
            for feat in features:
                geom = feat.get("geometry", {})
                gtype = geom.get("type")
                coords = geom.get("coordinates", [])

                if gtype == "Polygon":
                    poly_list = [coords]
                elif gtype == "MultiPolygon":
                    poly_list = coords
                else:
                    continue

                for p in poly_list:
                    if p and len(p[0]) >= 3:
                        self._polygons.append(PreparedPolygon(p[0]))

            self._is_loaded = bool(self._polygons)
            logger.info("Loaded %d land polygons for geographic context validation from %s", len(self._polygons), path)
        except Exception as exc:
            logger.error("Failed loading geographic context land dataset: %s", exc)
            self._is_loaded = False

    @staticmethod
    def _point_to_segment_distance_km(
        lon: float, lat: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0 and dy == 0.0:
            return haversine((lon, lat), (x1, y1))
        t = max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / (dx * dx + dy * dy)))
        return haversine((lon, lat), (x1 + t * dx, y1 + t * dy))

    def _distance_to_coast_km(self, lon: float, lat: float) -> float:
        if not self._polygons:
            return float("inf")

        min_d = float("inf")
        # Pass 1: Check polygons within a 5-degree bounding box pre-filter
        for p in self._polygons:
            if (p.min_x - 5.0) <= lon <= (p.max_x + 5.0) and (p.min_y - 5.0) <= lat <= (p.max_y + 5.0):
                if p.ext_arr is not None and len(p.ext_arr) > 1:
                    p1 = p.ext_arr[:-1]
                    p2 = p.ext_arr[1:]
                    d = p2 - p1
                    denom = np.sum(d**2, axis=1)
                    denom = np.where(denom == 0, 1e-9, denom)
                    u = np.clip(((lon - p1[:, 0]) * d[:, 0] + (lat - p1[:, 1]) * d[:, 1]) / denom, 0.0, 1.0)
                    closest = p1 + u[:, None] * d
                    deg_dist_sq = (closest[:, 0] - lon) ** 2 + (closest[:, 1] - lat) ** 2
                    top_k = min(len(closest), 15)
                    top_idx = np.argpartition(deg_dist_sq, top_k - 1)[:top_k]
                    for idx in top_idx:
                        d_km = haversine((lon, lat), (float(closest[idx, 0]), float(closest[idx, 1])))
                        if d_km < min_d:
                            min_d = d_km
                else:
                    ext = p.exterior
                    for i in range(len(ext) - 1):
                        d = self._point_to_segment_distance_km(
                            lon, lat, ext[i][0], ext[i][1], ext[i + 1][0], ext[i + 1][1]
                        )
                        if d < min_d:
                            min_d = d

        # Pass 2: Fallback across all polygons if remote/oceanic
        if min_d == float("inf"):
            for p in self._polygons:
                ext = p.exterior
                for i in range(len(ext) - 1):
                    d = self._point_to_segment_distance_km(
                        lon, lat, ext[i][0], ext[i][1], ext[i + 1][0], ext[i + 1][1]
                    )
                    if d < min_d:
                        min_d = d

        return min_d

    def _is_on_land(self, lon: float, lat: float) -> bool:
        for p in self._polygons:
            if p.min_x <= lon <= p.max_x and p.min_y <= lat <= p.max_y:
                if pnpoly(p.nvert, p.vx, p.vy, lon, lat):
                    return True
        return False

    def validate_candidate(
        self,
        lat: float,
        lon: float,
    ) -> GeographicContextResult:
        """
        Validate the geographic context for a detection at (lat, lon).
        
        Returns:
        - PLAUSIBLE_WATER: Marine / offshore target on water.
        - TERRESTRIAL_CLUTTER: High-confidence inland continental clutter.
        - AMBIGUOUS_COASTAL: Nearshore / harbor target within buffer boundary.
        - UNKNOWN: Land/water mask unavailable or unvalidated.
        """
        if not self._is_loaded:
            return GeographicContextResult(
                status=GeographicContextStatus.UNKNOWN,
                reason="Geographic land/water mask data unavailable; uncertainty unvalidated, candidate retained.",
                is_operational_vessel=True,
                data_source="None",
            )

        # Coordinate domain check
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return GeographicContextResult(
                status=GeographicContextStatus.UNKNOWN,
                reason=f"Coordinates out of bounds: lat={lat}, lon={lon}",
                is_operational_vessel=True,
                data_source=self.data_source_name,
            )

        is_land = self._is_on_land(lon, lat)
        dist_coast_km = self._distance_to_coast_km(lon, lat)

        # 1. Inland Continental Landmass -> TERRESTRIAL_CLUTTER
        if is_land:
            if dist_coast_km > self.coastal_ambiguity_buffer_km:
                return GeographicContextResult(
                    status=GeographicContextStatus.TERRESTRIAL_CLUTTER,
                    reason=(
                        f"Candidate is located {dist_coast_km:.1f} km inland from coastline on continental landmass; "
                        "flagged as terrestrial radar clutter."
                    ),
                    is_operational_vessel=False,
                    distance_to_coast_km=dist_coast_km,
                    is_on_land=True,
                    data_source=self.data_source_name,
                )
            else:
                return GeographicContextResult(
                    status=GeographicContextStatus.AMBIGUOUS_COASTAL,
                    reason=(
                        f"Candidate is on land but within coastal ambiguity buffer ({dist_coast_km:.1f} km from coastline); "
                        "potential port/docked vessel or coastal boundary uncertainty."
                    ),
                    is_operational_vessel=True,
                    distance_to_coast_km=dist_coast_km,
                    is_on_land=True,
                    data_source=self.data_source_name,
                )

        # 2. Water Surface
        if dist_coast_km <= self.coastal_ambiguity_buffer_km:
            return GeographicContextResult(
                status=GeographicContextStatus.AMBIGUOUS_COASTAL,
                reason=(
                    f"Candidate is in nearshore waters within coastal ambiguity buffer ({dist_coast_km:.1f} km from coastline); "
                    "plausible nearshore/harbor target."
                ),
                is_operational_vessel=True,
                distance_to_coast_km=dist_coast_km,
                is_on_land=False,
                data_source=self.data_source_name,
            )

        return GeographicContextResult(
            status=GeographicContextStatus.PLAUSIBLE_WATER,
            reason=f"Candidate is located {dist_coast_km:.1f} km offshore in open navigable waters.",
            is_operational_vessel=True,
            distance_to_coast_km=dist_coast_km,
            is_on_land=False,
            data_source=self.data_source_name,
        )
