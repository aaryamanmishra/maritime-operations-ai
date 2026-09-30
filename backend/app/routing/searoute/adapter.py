import math
from typing import List, Tuple, Dict, Any
import searoute as sr
from app.domain.weather_routing.models import GeoCoordinate


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance between two points in nautical miles."""
    r_nm = 3440.065
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_nm * c


def calculate_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate forward initial azimuth (bearing) from (lat1, lon1) to (lat2, lon2) in [0, 360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return bearing


class SeaRouteAdapter:
    """
    Adapter for the SeaRoute open maritime network (Eurostat / SeaRoute).
    Calculates maritime geometry avoiding land and returns detailed segment breakdowns.
    """
    def __init__(self):
        pass

    def calculate_base_maritime_route(
        self,
        origin: GeoCoordinate,
        destination: GeoCoordinate,
    ) -> Dict[str, Any]:
        """
        Query the SeaRoute maritime network for water-only passage.
        Coordinates are passed as [lon, lat] per GeoJSON convention.
        """
        orig_pt = [origin.longitude, origin.latitude]
        dest_pt = [destination.longitude, destination.latitude]

        try:
            route_geojson = sr.searoute(orig_pt, dest_pt, units="naut")
        except Exception as exc:
            raise RuntimeError(f"SeaRoute pathfinding failed: {exc}") from exc

        coords = route_geojson["geometry"]["coordinates"]  # [[lon, lat], ...]
        if not coords or len(coords) < 2:
            raise ValueError("SeaRoute returned insufficient coordinates for route")

        # Densify segments if spacing > 60 nm, or keep reasonable spacing
        densified_coords = self._sample_and_densify(coords, max_segment_nm=50.0)

        # Build detailed segment metadata
        segments = []
        total_dist_nm = 0.0

        for i in range(len(densified_coords) - 1):
            p1_lon, p1_lat = densified_coords[i]
            p2_lon, p2_lat = densified_coords[i + 1]

            seg_dist = haversine_distance_nm(p1_lat, p1_lon, p2_lat, p2_lon)
            bearing = calculate_bearing_deg(p1_lat, p1_lon, p2_lat, p2_lon)
            total_dist_nm += seg_dist

            segments.append({
                "segment_index": i,
                "start": (p1_lat, p1_lon),
                "end": (p2_lat, p2_lon),
                "distance_nm": round(seg_dist, 2),
                "bearing_deg": round(bearing, 1),
            })

        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": densified_coords,
            },
            "properties": {
                "total_distance_nm": round(total_dist_nm, 2),
                "segment_count": len(segments),
            },
            "segments": segments,
        }

    def _sample_and_densify(
        self, coords: List[List[float]], max_segment_nm: float = 50.0
    ) -> List[List[float]]:
        """Ensure segments are not too long for weather sampling."""
        densified = [coords[0]]
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            dist = haversine_distance_nm(lat1, lon1, lat2, lon2)
            
            if dist > max_segment_nm:
                num_sub = int(math.ceil(dist / max_segment_nm))
                for s in range(1, num_sub):
                    frac = s / float(num_sub)
                    interp_lat = lat1 + frac * (lat2 - lat1)
                    interp_lon = lon1 + frac * (lon2 - lon1)
                    densified.append([interp_lon, interp_lat])
            densified.append([lon2, lat2])
        return densified

    def generate_alternative_corridors(
        self,
        base_route: Dict[str, Any],
        lateral_offset_nm: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate corridor routes for weather avoidance.
        Evaluates lateral offsets across open-water waypoints via SeaRoute.
        """
        coords = base_route["geometry"]["coordinates"]
        if len(coords) < 4:
            return []

        candidates = []
        # Try a northward and southward intermediate waypoint detour
        mid_idx = len(coords) // 2
        mid_lon, mid_lat = coords[mid_idx]

        for sign in [1.0, -1.0]:
            # Approximate lateral displacement perpendicular to local bearing
            alt_lat = mid_lat + (sign * (lateral_offset_nm / 60.0))
            alt_lon = mid_lon + (sign * (lateral_offset_nm / (60.0 * max(math.cos(math.radians(mid_lat)), 0.1))))
            
            try:
                # Leg 1: origin to intermediate
                orig = GeoCoordinate(latitude=coords[0][1], longitude=coords[0][0])
                mid_pt = GeoCoordinate(latitude=alt_lat, longitude=alt_lon)
                dest = GeoCoordinate(latitude=coords[-1][1], longitude=coords[-1][0])

                leg1 = self.calculate_base_maritime_route(orig, mid_pt)
                leg2 = self.calculate_base_maritime_route(mid_pt, dest)

                # Combine coordinates
                combined_coords = leg1["geometry"]["coordinates"] + leg2["geometry"]["coordinates"][1:]
                tot_dist = leg1["properties"]["total_distance_nm"] + leg2["properties"]["total_distance_nm"]
                
                # Combine segments
                comb_segments = []
                idx = 0
                for s in leg1["segments"] + leg2["segments"]:
                    s_copy = dict(s)
                    s_copy["segment_index"] = idx
                    comb_segments.append(s_copy)
                    idx += 1

                candidates.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": combined_coords},
                    "properties": {"total_distance_nm": round(tot_dist, 2), "segment_count": len(comb_segments)},
                    "segments": comb_segments,
                    "label": f"detour_{'north' if sign > 0 else 'south'}",
                })
            except Exception:
                # Discard detours that cross land or fail
                pass

        return candidates
