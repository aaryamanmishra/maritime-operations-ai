from typing import Any


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
        corner_coords: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
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
    def pixel_to_lonlat_grid(
        x: float,
        y: float,
        grid_points: list[dict[str, Any]],
        x_offset: float = 0.0,
        y_offset: float = 0.0,
    ) -> tuple[float, float]:
        """
        Convert pixel (x, y) with optional chip offset to (lon, lat) using the
        product's actual Geolocation Grid Points (GCPs) via piecewise bilinear interpolation.
        Eliminates non-linear slant-range and Earth-curvature geometric distortions
        observed in global 4-corner affine approximations across wide radar swaths.
        """
        if not grid_points:
            raise ValueError("grid_points list cannot be empty")

        import bisect

        abs_x = max(0.0, float(x + x_offset))
        abs_y = max(0.0, float(y + y_offset))

        # Group points by line
        lines_dict: dict[float, list[dict[str, Any]]] = {}
        for p in grid_points:
            l = float(p.get("line", 0.0))
            lines_dict.setdefault(l, []).append(p)

        sorted_lines = sorted(lines_dict.keys())
        for l in sorted_lines:
            lines_dict[l].sort(key=lambda pt: float(pt.get("pixel", 0.0)))

        # Find enclosing line interval [l0, l1]
        if abs_y <= sorted_lines[0]:
            y0_idx, y1_idx = 0, min(1, len(sorted_lines) - 1)
        elif abs_y >= sorted_lines[-1]:
            y0_idx, y1_idx = max(0, len(sorted_lines) - 2), len(sorted_lines) - 1
        else:
            idx = bisect.bisect_right(sorted_lines, abs_y) - 1
            y0_idx, y1_idx = idx, idx + 1

        l0, l1 = sorted_lines[y0_idx], sorted_lines[y1_idx]
        v = 0.0 if l1 == l0 else max(0.0, min(1.0, (abs_y - l0) / (l1 - l0)))

        row0 = lines_dict[l0]
        row1 = lines_dict[l1]

        # Find enclosing pixel interval in row0
        pixels0 = [float(pt.get("pixel", 0.0)) for pt in row0]
        if abs_x <= pixels0[0]:
            x0_idx, x1_idx = 0, min(1, len(pixels0) - 1)
        elif abs_x >= pixels0[-1]:
            x0_idx, x1_idx = max(0, len(pixels0) - 2), len(pixels0) - 1
        else:
            idx = bisect.bisect_right(pixels0, abs_x) - 1
            x0_idx, x1_idx = idx, idx + 1

        px0, px1 = pixels0[x0_idx], pixels0[x1_idx]
        u = 0.0 if px1 == px0 else max(0.0, min(1.0, (abs_x - px0) / (px1 - px0)))

        p_tl = row0[x0_idx]
        p_tr = row0[x1_idx]
        p_bl = row1[min(x0_idx, len(row1) - 1)]
        p_br = row1[min(x1_idx, len(row1) - 1)]

        lon = (
            (1.0 - u) * (1.0 - v) * float(p_tl["longitude"])
            + u * (1.0 - v) * float(p_tr["longitude"])
            + (1.0 - u) * v * float(p_bl["longitude"])
            + u * v * float(p_br["longitude"])
        )
        lat = (
            (1.0 - u) * (1.0 - v) * float(p_tl["latitude"])
            + u * (1.0 - v) * float(p_tr["latitude"])
            + (1.0 - u) * v * float(p_bl["latitude"])
            + u * v * float(p_br["latitude"])
        )

        return round(lon, 6), round(lat, 6)

    @staticmethod
    def extract_corners_from_footprint(
        footprint_geojson: dict[str, Any],
    ) -> dict[str, tuple[float, float]]:
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

    @staticmethod
    def calculate_vessel_length(
        bbox: list[float] | tuple[float, float, float, float],
        modality: str = "GRD",
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Estimate vessel length in meters based on bounding box dimensions and sensor geometry.

        - GRD (Ground Range Detected):
          Resampled to square ground range resolution cells, nominally 10.0 m x 10.0 m (High-Res GRDH).
          Length = sqrt((dx * 10.0)^2 + (dy * 10.0)^2)

        - SLC (Single Look Complex, IW mode):
          Native slant-range pixel spacing is Delta_r = c / (2 * fs) ~= 3.38 m (range sampling freq fs = 44.33 MHz).
          Native azimuth pixel spacing is Delta_a ~= 14.0 m (based on orbital velocity v_orb ~= 7500 m/s and PRF ~= 1717 Hz).
          If metadata provides exact 'sar:resolution_range' or 'sar:resolution_azimuth', those are used.
          Ground-range projection: dx_ground = dx_slant / sin(incidence_angle) if incidence_angle is available.
          Length = sqrt((dx * range_spacing)^2 + (dy * azimuth_spacing)^2)
        """
        xmin, ymin, xmax, ymax = bbox[:4]
        dx = max(0.0, float(xmax - xmin))
        dy = max(0.0, float(ymax - ymin))

        mod = modality.upper()
        metadata = metadata or {}

        if mod == "GRD":
            # Standard S1 GRDH nominal pixel spacing: 10.0m x 10.0m
            px_range = float(metadata.get("sar:resolution_range") or 10.0)
            px_azimuth = float(metadata.get("sar:resolution_azimuth") or 10.0)
            length_m = ((dx * px_range) ** 2 + (dy * px_azimuth) ** 2) ** 0.5
            return round(length_m, 1)
        elif mod == "SLC":
            # S1 IW SLC slant-range and azimuth pixel spacing
            # - Range sampling rate varies by subswath:
            #   IW1: fs = 64.35 MHz -> rangePixelSpacing = 2.33 m slant-range
            #   IW2/IW3: fs = 44.33 MHz -> rangePixelSpacing = 3.38 m slant-range
            # Respect product-provided metadata if present.
            subswath = str(metadata.get("sar:subswath") or metadata.get("swath") or "").upper()
            if "rangePixelSpacing" in metadata or "sar:range_pixel_spacing" in metadata:
                px_range = float(metadata.get("rangePixelSpacing") or metadata.get("sar:range_pixel_spacing"))
            elif subswath == "IW1":
                px_range = 2.329562
            else:
                px_range = float(metadata.get("sar:resolution_range") or 3.38)

            if "azimuthPixelSpacing" in metadata or "sar:azimuth_pixel_spacing" in metadata:
                px_azimuth = float(metadata.get("azimuthPixelSpacing") or metadata.get("sar:azimuth_pixel_spacing"))
            else:
                px_azimuth = float(metadata.get("sar:resolution_azimuth") or 14.0)

            # If incidence angle is provided, convert slant range to ground range
            inc_deg = metadata.get("sar:incidence_angle") or metadata.get("incidenceAngle")
            if inc_deg is not None:
                import math
                inc_rad = math.radians(float(inc_deg))
                if math.sin(inc_rad) > 0.1:
                    px_range = px_range / math.sin(inc_rad)

            length_m = ((dx * px_range) ** 2 + (dy * px_azimuth) ** 2) ** 0.5
            return round(length_m, 1)
        else:
            # Fallback
            diag_px = (dx ** 2 + dy ** 2) ** 0.5
            return round(diag_px * 10.0, 1)
