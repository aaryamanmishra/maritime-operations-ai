import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


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

        abs_x = float(x + x_offset)
        abs_y = float(y + y_offset)

        # Group points by line
        lines_dict: dict[float, list[dict[str, Any]]] = {}
        for p in grid_points:
            line_val = float(p.get("line", 0.0))
            lines_dict.setdefault(line_val, []).append(p)

        sorted_lines = sorted(lines_dict.keys())
        for line_val in sorted_lines:
            lines_dict[line_val].sort(key=lambda pt: float(pt.get("pixel", 0.0)))

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
            x0_idx_0, x1_idx_0 = 0, min(1, len(pixels0) - 1)
        elif abs_x >= pixels0[-1]:
            x0_idx_0, x1_idx_0 = max(0, len(pixels0) - 2), len(pixels0) - 1
        else:
            idx = bisect.bisect_right(pixels0, abs_x) - 1
            x0_idx_0, x1_idx_0 = idx, idx + 1

        px0_0, px1_0 = pixels0[x0_idx_0], pixels0[x1_idx_0]
        u0 = 0.0 if px1_0 == px0_0 else max(0.0, min(1.0, (abs_x - px0_0) / (px1_0 - px0_0)))

        # Find enclosing pixel interval in row1
        pixels1 = [float(pt.get("pixel", 0.0)) for pt in row1]
        if abs_x <= pixels1[0]:
            x0_idx_1, x1_idx_1 = 0, min(1, len(pixels1) - 1)
        elif abs_x >= pixels1[-1]:
            x0_idx_1, x1_idx_1 = max(0, len(pixels1) - 2), len(pixels1) - 1
        else:
            idx = bisect.bisect_right(pixels1, abs_x) - 1
            x0_idx_1, x1_idx_1 = idx, idx + 1

        px0_1, px1_1 = pixels1[x0_idx_1], pixels1[x1_idx_1]
        u1 = 0.0 if px1_1 == px0_1 else max(0.0, min(1.0, (abs_x - px0_1) / (px1_1 - px0_1)))

        p_tl = row0[x0_idx_0]
        p_tr = row0[x1_idx_0]
        p_bl = row1[x0_idx_1]
        p_br = row1[x1_idx_1]

        # Longitude interpolation with antimeridian (±180 deg) wrapping support
        def interp_lon(l0_deg: float, l1_deg: float, weight: float) -> float:
            d = (l1_deg - l0_deg + 180.0) % 360.0 - 180.0
            return (l0_deg + weight * d + 180.0) % 360.0 - 180.0

        lon_top = interp_lon(float(p_tl["longitude"]), float(p_tr["longitude"]), u0)
        lat_top = (1.0 - u0) * float(p_tl["latitude"]) + u0 * float(p_tr["latitude"])

        lon_bot = interp_lon(float(p_bl["longitude"]), float(p_br["longitude"]), u1)
        lat_bot = (1.0 - u1) * float(p_bl["latitude"]) + u1 * float(p_br["latitude"])

        lon = interp_lon(lon_top, lon_bot, v)
        lat = (1.0 - v) * lat_top + v * lat_bot

        return round(lon, 6), round(lat, 6)

    @staticmethod
    def pixel_to_lonlat_affine(
        x: float,
        y: float,
        geotransform: list[float] | tuple[float, ...],
        crs: str | None = None,
    ) -> tuple[float, float]:
        """
        Transform pixel (x, y) to (longitude, latitude) in WGS84 EPSG:4326 using a 6-element
        affine GeoTransform:
          X_geo = gt[0] + x * gt[1] + y * gt[2]
          Y_geo = gt[3] + x * gt[4] + y * gt[5]
        """
        if not geotransform or len(geotransform) < 6:
            raise ValueError(f"Invalid geotransform: expected 6 elements, got {geotransform}")

        gt = geotransform
        x_geo = float(gt[0]) + float(x) * float(gt[1]) + float(y) * float(gt[2])
        y_geo = float(gt[3]) + float(x) * float(gt[4]) + float(y) * float(gt[5])

        if not crs or "4326" in crs or "WGS 84" in crs or "WGS_1984" in crs:
            return round(float(x_geo), 6), round(float(y_geo), 6)

        try:
            from osgeo import osr
            src_srs = osr.SpatialReference()
            src_srs.SetFromUserInput(crs)
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(4326)
            dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            ct = osr.CoordinateTransformation(src_srs, dst_srs)
            pt = ct.TransformPoint(float(x_geo), float(y_geo))
            return round(float(pt[0]), 6), round(float(pt[1]), 6)
        except Exception as e:
            logger.warning("CRS reprojection to EPSG:4326 failed (%s), returning raw affine coordinates", e)
            return round(float(x_geo), 6), round(float(y_geo), 6)

    @classmethod
    def pixel_to_geographic(
        cls,
        x: float,
        y: float,
        metadata: dict[str, Any] | None = None,
        parent_width: float | None = None,
        parent_height: float | None = None,
        corner_coords: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
        """
        Unified georeferencing entry point converting image pixel coordinates to WGS84 (lon, lat).

        Prioritizes:
        1. Authoritative Geolocation Grid Points (GCPs): piecewise bilinear interpolation
           accounting for downsampling (scale_x, scale_y) and cropping (crop_x_offset, crop_y_offset).
        2. Authoritative Affine GeoTransform & CRS: exact raster affine projection.
        3. Approximate 4-corner Bilinear Interpolation: strictly as documented fallback
           when corner_coords is explicitly provided and authoritative data is absent.
        4. Explicit Failure: raises RuntimeError when no georeferencing data is available.
        """
        meta = metadata or {}

        # 1. Authoritative GCP Grid (Standard Sentinel-1 GRD & SLC burst chips)
        grid_points = meta.get("geolocation_grid_points")
        if grid_points and isinstance(grid_points, list) and len(grid_points) >= 4:
            # Determine scale factors if image was resized/downsampled
            if (
                parent_width
                and meta.get("original_width")
                and int(parent_width) == int(meta["original_width"])
            ):
                scale_x = 1.0
            else:
                scale_x = float(meta.get("scale_x") or 1.0)
                if scale_x == 1.0 and parent_width and meta.get("original_width"):
                    scale_x = float(meta["original_width"]) / float(parent_width)

            if (
                parent_height
                and meta.get("original_height")
                and int(parent_height) == int(meta["original_height"])
            ):
                scale_y = 1.0
            else:
                scale_y = float(meta.get("scale_y") or 1.0)
                if scale_y == 1.0 and parent_height and meta.get("original_height"):
                    scale_y = float(meta["original_height"]) / float(parent_height)

            crop_x = float(meta.get("crop_x_offset") or 0.0)
            crop_y = float(meta.get("crop_y_offset") or 0.0)

            orig_x = float(x) * scale_x + crop_x
            orig_y = float(y) * scale_y + crop_y

            return cls.pixel_to_lonlat_grid(orig_x, orig_y, grid_points)

        # 2. Authoritative Affine GeoTransform
        geotransform = meta.get("geotransform")
        if geotransform and len(geotransform) >= 6:
            if (
                parent_width
                and meta.get("original_width")
                and int(parent_width) == int(meta["original_width"])
            ):
                scale_x = 1.0
            else:
                scale_x = float(meta.get("scale_x") or 1.0)
                if scale_x == 1.0 and parent_width and meta.get("original_width"):
                    scale_x = float(meta["original_width"]) / float(parent_width)

            if (
                parent_height
                and meta.get("original_height")
                and int(parent_height) == int(meta["original_height"])
            ):
                scale_y = 1.0
            else:
                scale_y = float(meta.get("scale_y") or 1.0)
                if scale_y == 1.0 and parent_height and meta.get("original_height"):
                    scale_y = float(meta["original_height"]) / float(parent_height)

            crop_x = float(meta.get("crop_x_offset") or 0.0)
            crop_y = float(meta.get("crop_y_offset") or 0.0)

            orig_x = float(x) * scale_x + crop_x
            orig_y = float(y) * scale_y + crop_y

            return cls.pixel_to_lonlat_affine(orig_x, orig_y, geotransform, meta.get("crs"))

        # 3. Bilinear fallback with corner_coords
        if corner_coords and parent_width and parent_height:
            return cls.pixel_to_lonlat_bilinear(float(x), float(y), float(parent_width), float(parent_height), corner_coords)

        # 4. Explicit Error: Never fabricate coordinates
        raise RuntimeError(
            "No valid georeferencing metadata (geolocation_grid_points, geotransform, "
            "or corner_coords) available for pixel coordinate transformation."
        )

    @staticmethod
    def extract_corners_from_footprint(
        footprint_geojson: dict[str, Any],
        orbit_pass: str | None = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Extract 4 corner coordinates from a GeoJSON Polygon footprint geometry.
        WARNING: Footprint boundary vertices provide only an outer boundary approximation and
        cannot account for satellite look angle, radar projection, or internal distortions.
        Authoritative GCP grids or GeoTransforms must ALWAYS be preferred.

        Handles both closed and unclosed coordinate rings.
        If orbit_pass is 'DESCENDING', accounts for Sentinel-1 descending orbit geometry
        (satellite moving south, looking west).
        Otherwise sorts vertices geometrically into Top-Left, Top-Right, Bottom-Right, Bottom-Left
        based on geographic orientation.
        """
        coords = footprint_geojson.get("coordinates", [])
        if not coords or not coords[0]:
            raise ValueError("Invalid footprint GeoJSON: missing coordinates")

        raw_ring = coords[0]
        # Deduplicate closing point if present
        ring = []
        for pt in raw_ring:
            pt_tup = (float(pt[0]), float(pt[1]))
            if not ring or ring[0] != pt_tup or len(ring) < 4:
                ring.append(pt_tup)
        if len(ring) > 4 and ring[0] == ring[-1]:
            ring = ring[:-1]

        if len(ring) < 4:
            raise ValueError(f"Expected at least 4 boundary coordinates, got {len(ring)}")

        # 4 unique vertices
        v = ring[:4]

        orbit_pass_upper = (orbit_pass or "").upper()
        if orbit_pass_upper == "DESCENDING" and len(raw_ring) >= 5:
            # Standard ESA STAC Sentinel-1 Descending GeoJSON rings are ordered:
            # P0: South-East (Raster Bottom-Left)
            # P1: North-East (Raster Top-Left)
            # P2: North-West (Raster Top-Right)
            # P3: South-West (Raster Bottom-Right)
            # P4: Closing point (= P0)
            return {
                "top_left": (float(raw_ring[1][0]), float(raw_ring[1][1])),
                "top_right": (float(raw_ring[2][0]), float(raw_ring[2][1])),
                "bottom_right": (float(raw_ring[3][0]), float(raw_ring[3][1])),
                "bottom_left": (float(raw_ring[0][0]), float(raw_ring[0][1])),
            }

        # Geometric sorting by latitude and longitude:
        # Top two vertices have largest latitude; bottom two have smallest latitude.
        sorted_by_lat = sorted(v, key=lambda p: p[1], reverse=True)
        top_two = sorted(sorted_by_lat[:2], key=lambda p: p[0])
        bottom_two = sorted(sorted_by_lat[2:], key=lambda p: p[0])

        tl = top_two[0]  # higher lat, lower lon
        tr = top_two[1]  # higher lat, higher lon
        bl = bottom_two[0]  # lower lat, lower lon
        br = bottom_two[1]  # lower lat, higher lon

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
                inc_rad = math.radians(float(inc_deg))
                if math.sin(inc_rad) > 0.1:
                    px_range = px_range / math.sin(inc_rad)

            length_m = ((dx * px_range) ** 2 + (dy * px_azimuth) ** 2) ** 0.5
            return round(length_m, 1)
        else:
            # Fallback
            diag_px = (dx ** 2 + dy ** 2) ** 0.5
            return round(diag_px * 10.0, 1)
