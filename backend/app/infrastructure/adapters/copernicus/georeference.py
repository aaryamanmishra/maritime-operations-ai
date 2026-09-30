import math
from typing import Dict, List, Optional, Tuple, Any


class SARGeoreferencer:
    """
    Georeferencing utilities converting SAR image pixel coordinates (x, y)
    to WGS84 geographic coordinates (longitude, latitude).
    """

    @staticmethod
    def pixel_to_lonlat_bilinear(
        x: float,
        y: float,
        width: float,
        height: float,
        corner_coords: Dict[str, Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Convert pixel (x, y) within a raster of dimensions (width, height) to (lon, lat)
        via bilinear interpolation over 4 corner coordinates.
        corner_coords format: {
            "top_left": (lon, lat),
            "top_right": (lon, lat),
            "bottom_right": (lon, lat),
            "bottom_left": (lon, lat),
        }
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: width={width}, height={height}")

        u = max(0.0, min(1.0, x / width))
        v = max(0.0, min(1.0, y / height))

        tl_lon, tl_lat = corner_coords["top_left"]
        tr_lon, tr_lat = corner_coords["top_right"]
        br_lon, br_lat = corner_coords["bottom_right"]
        bl_lon, bl_lat = corner_coords["bottom_left"]

        # Bilinear interpolation
        lon = (
            (1.0 - u) * (1.0 - v) * tl_lon
            + u * (1.0 - v) * tr_lon
            + u * v * br_lon
            + (1.0 - u) * v * bl_lon
        )
        lat = (
            (1.0 - u) * (1.0 - v) * tl_lat
            + u * (1.0 - v) * tr_lat
            + u * v * br_lat
            + (1.0 - u) * v * bl_lat
        )

        return round(lon, 6), round(lat, 6)

    @staticmethod
    def extract_corners_from_footprint(
        footprint_geojson: Dict[str, Any],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Extract 4 corners from a GeoJSON Polygon geometry.
        """
        coords = footprint_geojson.get("coordinates", [])
        if not coords or not coords[0]:
            raise ValueError("Invalid footprint GeoJSON: missing coordinates")

        ring = coords[0]
        # Ring has at least 4 or 5 points (closed loop)
        if len(ring) < 4:
            raise ValueError(f"Expected at least 4 boundary coordinates, got {len(ring)}")

        # Sentinel-1 STAC footprints are typically oriented TL, TR, BR, BL, TL
        tl = (float(ring[0][0]), float(ring[0][1]))
        tr = (float(ring[1][0]), float(ring[1][1]))
        br = (float(ring[2][0]), float(ring[2][1]))
        bl = (float(ring[3][0]), float(ring[3][1]))

        return {
            "top_left": tl,
            "top_right": tr,
            "bottom_right": br,
            "bottom_left": bl,
        }
