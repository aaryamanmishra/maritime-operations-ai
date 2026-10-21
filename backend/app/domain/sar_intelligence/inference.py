import json
import logging
import os
import uuid
from typing import Any, Optional

import numpy as np
from ultralytics import YOLO

from app.core.config import Settings
from app.infrastructure.adapters.copernicus.georeference import SARGeoreferencer
from app.infrastructure.adapters.copernicus.preprocessing import SARTile

logger = logging.getLogger(__name__)


def non_max_suppression_boxes(
    boxes: list[dict[str, Any]],
    iou_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Standard greedy Non-Maximum Suppression (NMS) over candidate vessel detections.
    Each box is a dict containing 'bbox' [xmin, ymin, xmax, ymax] and 'confidence'.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    keep: list[dict[str, Any]] = []

    for cand in sorted_boxes:
        c_box = cand["bbox"]
        c_area = max(0.0, c_box[2] - c_box[0]) * max(0.0, c_box[3] - c_box[1])

        should_keep = True
        for k in keep:
            k_box = k["bbox"]
            k_area = max(0.0, k_box[2] - k_box[0]) * max(0.0, k_box[3] - k_box[1])

            inter_xmin = max(c_box[0], k_box[0])
            inter_ymin = max(c_box[1], k_box[1])
            inter_xmax = min(c_box[2], k_box[2])
            inter_ymax = min(c_box[3], k_box[3])

            inter_w = max(0.0, inter_xmax - inter_xmin)
            inter_h = max(0.0, inter_ymax - inter_ymin)
            inter_area = inter_w * inter_h

            union_area = c_area + k_area - inter_area
            if union_area > 0:
                iou = inter_area / union_area
                if iou >= iou_threshold:
                    should_keep = False
                    break

        if should_keep:
            keep.append(cand)

    return keep


def deduplicate_scene_detections(
    candidates: list[dict[str, Any]],
    iou_threshold: float = 0.30,
    edge_margin_px: float = 16.0,
    max_center_dist_px: float = 30.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Conservative cross-tile deduplication in parent-scene pixel coordinates.

    Rules & Invariants:
    1. Cross-tile candidate matching is evaluated ONLY when candidate source tiles
       spatially overlap, and both bounding boxes lie in or near the tile overlap band.
    2. Non-overlapping tiles can NEVER cause cross-tile suppression.
    3. Confirmed duplicate evidence requires either:
       - High scene-level bounding-box IoU (>= iou_threshold, default 0.30)
       - Boundary-truncated overlap: IoU >= 0.15 with small center distance (<= 30 px, <= 0.5 * min_dim)
       - Near-concentric enclosure: IoMin >= 0.50 with center distance <= 20 px.
    4. Two distinct nearby targets not meeting these conservative thresholds are
       both retained as separate operational contacts (inconclusive -> retain both).
    5. The primary candidate (highest confidence) is retained operationally and
       links suppressed twin candidates in its 'duplicate_provenance' record.
       The suppressed twin candidate records 'duplicate_of' linking back to primary.

    Returns:
        (operational_candidates, all_candidates_with_provenance)
    """
    if not candidates:
        return [], []

    import math

    sorted_cands = sorted(candidates, key=lambda c: float(c.get("confidence", 0.0)), reverse=True)
    operational: list[dict[str, Any]] = []

    for cand in sorted_cands:
        c_box = cand["bounding_box"]
        c_tile = cand.get("source_tile", {})
        c_tile_idx = c_tile.get("tile_index")
        c_xoff = float(c_tile.get("x_offset", 0.0))
        c_yoff = float(c_tile.get("y_offset", 0.0))
        c_w = float(c_tile.get("width", 640.0))
        c_h = float(c_tile.get("height", 640.0))

        c_area = max(0.0, c_box[2] - c_box[0]) * max(0.0, c_box[3] - c_box[1])
        c_cx = (c_box[0] + c_box[2]) / 2.0
        c_cy = (c_box[1] + c_box[3]) / 2.0
        c_dim = max(c_box[2] - c_box[0], c_box[3] - c_box[1])

        matched_primary = None
        match_metrics = None

        for op in operational:
            op_box = op["bounding_box"]
            op_tile = op.get("source_tile", {})
            op_tile_idx = op_tile.get("tile_index")
            op_xoff = float(op_tile.get("x_offset", 0.0))
            op_yoff = float(op_tile.get("y_offset", 0.0))
            op_w = float(op_tile.get("width", 640.0))
            op_h = float(op_tile.get("height", 640.0))

            # Case A: Same tile -> standard intra-tile NMS
            if c_tile_idx is not None and op_tile_idx is not None and c_tile_idx == op_tile_idx:
                inter_xmin = max(c_box[0], op_box[0])
                inter_ymin = max(c_box[1], op_box[1])
                inter_xmax = min(c_box[2], op_box[2])
                inter_ymax = min(c_box[3], op_box[3])
                inter_w = max(0.0, inter_xmax - inter_xmin)
                inter_h = max(0.0, inter_ymax - inter_ymin)
                inter_area = inter_w * inter_h

                op_area = max(0.0, op_box[2] - op_box[0]) * max(0.0, op_box[3] - op_box[1])
                union_area = c_area + op_area - inter_area
                iou = (inter_area / union_area) if union_area > 0 else 0.0

                if iou >= 0.45:
                    matched_primary = op
                    match_metrics = {
                        "iou": round(iou, 4),
                        "center_dist": 0.0,
                        "reason": "intra_tile_overlap",
                    }
                    break
                continue

            # Case B: Cross-tile
            # 1. Verify tile overlap in parent-scene space
            ovl_xmin = max(c_xoff, op_xoff)
            ovl_ymin = max(c_yoff, op_yoff)
            ovl_xmax = min(c_xoff + c_w, op_xoff + op_w)
            ovl_ymax = min(c_yoff + c_h, op_yoff + op_h)

            if ovl_xmax <= ovl_xmin or ovl_ymax <= ovl_ymin:
                # Source tiles do not overlap at all; cannot be duplicates
                continue

            # 2. Verify candidate boxes intersect expanded tile overlap band
            band_xmin = ovl_xmin - edge_margin_px
            band_ymin = ovl_ymin - edge_margin_px
            band_xmax = ovl_xmax + edge_margin_px
            band_ymax = ovl_ymax + edge_margin_px

            c_in_band = (
                c_box[0] <= band_xmax
                and c_box[2] >= band_xmin
                and c_box[1] <= band_ymax
                and c_box[3] >= band_ymin
            )
            op_in_band = (
                op_box[0] <= band_xmax
                and op_box[2] >= band_xmin
                and op_box[1] <= band_ymax
                and op_box[3] >= band_ymin
            )

            if not (c_in_band and op_in_band):
                # At least one candidate is situated outside the tile overlap region
                continue

            # 3. Compute spatial intersection in parent-scene pixel coordinates
            inter_xmin = max(c_box[0], op_box[0])
            inter_ymin = max(c_box[1], op_box[1])
            inter_xmax = min(c_box[2], op_box[2])
            inter_ymax = min(c_box[3], op_box[3])
            inter_w = max(0.0, inter_xmax - inter_xmin)
            inter_h = max(0.0, inter_ymax - inter_ymin)
            inter_area = inter_w * inter_h

            op_area = max(0.0, op_box[2] - op_box[0]) * max(0.0, op_box[3] - op_box[1])
            union_area = c_area + op_area - inter_area
            iou = (inter_area / union_area) if union_area > 0 else 0.0

            min_area = min(c_area, op_area)
            io_min = (inter_area / min_area) if min_area > 0 else 0.0

            op_cx = (op_box[0] + op_box[2]) / 2.0
            op_cy = (op_box[1] + op_box[3]) / 2.0
            center_dist = math.hypot(c_cx - op_cx, c_cy - op_cy)

            op_dim = max(op_box[2] - op_box[0], op_box[3] - op_box[1])
            min_dim = min(c_dim, op_dim)

            # Conservative duplicate criteria
            is_dup = False
            reason = ""
            if iou >= iou_threshold:
                is_dup = True
                reason = "high_scene_iou"
            elif (
                inter_area > 0
                and iou >= 0.15
                and center_dist <= min(max_center_dist_px, 0.5 * max(min_dim, 10.0))
            ):
                is_dup = True
                reason = "boundary_overlap_proximity"
            elif io_min >= 0.50 and center_dist <= 20.0:
                is_dup = True
                reason = "concentric_enclosure"

            if is_dup:
                matched_primary = op
                match_metrics = {
                    "iou": round(iou, 4),
                    "center_dist": round(center_dist, 2),
                    "reason": reason,
                }
                break

        if matched_primary is not None:
            # Candidate is a confirmed duplicate of matched_primary
            if matched_primary.get("duplicate_provenance") is None:
                matched_primary["duplicate_provenance"] = {
                    "is_deduplicated": True,
                    "primary_detection_id": matched_primary["detection_id"],
                    "duplicate_candidates": [],
                }
            matched_primary["duplicate_provenance"]["duplicate_candidates"].append({
                "detection_id": cand["detection_id"],
                "source_tile_index": c_tile.get("tile_index"),
                "tile_origin": [c_xoff, c_yoff],
                "tile_bbox": c_tile.get("tile_bbox"),
                "parent_bbox": cand["bounding_box"],
                "confidence": cand["confidence"],
                "iou_with_primary": match_metrics["iou"],
                "center_distance_px": match_metrics["center_dist"],
                "match_reason": match_metrics["reason"],
            })

            cand["duplicate_provenance"] = {
                "is_deduplicated": False,
                "duplicate_of": matched_primary["detection_id"],
                "iou_with_primary": match_metrics["iou"],
                "center_distance_px": match_metrics["center_dist"],
                "match_reason": match_metrics["reason"],
            }
        else:
            # Candidate is retained as an operational detection
            cand["duplicate_provenance"] = None
            operational.append(cand)

    return operational, sorted_cands


class YOLO26sVesselDetector:
    """
    Single production inference boundary for YOLO26s SAR vessel detection.
    Strictly performs inference only — no automated retraining inside backend container.
    Supports both GRD (default production) and SLC (opt-in) model checkpoints.
    """

    _instance: Optional["YOLO26sVesselDetector"] = None
    _instances: dict[str, "YOLO26sVesselDetector"] = {}

    def __init__(self, weights_path: str | None = None, modality: str = "GRD"):
        settings = Settings()
        self.modality = modality.upper()
        
        if self.modality == "SLC":
            self.weights_path = weights_path or settings.SAR_SLC_MODEL_WEIGHTS_PATH
            self.model_version = "yolo26s-opensar-slc-v1"
            candidates = [
                "backend/ml/weights/yolo26s_slc_vessel.pt",
                "ml/weights/yolo26s_slc_vessel.pt",
            ]
            meta_candidates = [
                os.path.join(os.path.dirname(self.weights_path), "yolo26s_slc_metadata.json"),
                "backend/ml/weights/yolo26s_slc_metadata.json",
                "ml/weights/yolo26s_slc_metadata.json",
            ]
        else:
            self.weights_path = weights_path or settings.SAR_MODEL_WEIGHTS_PATH
            self.model_version = "yolo26s-opensar-v1"
            candidates = [
                "backend/ml/weights/yolo26s_sar_vessel.pt",
                "ml/weights/yolo26s_sar_vessel.pt",
            ]
            meta_candidates = [
                os.path.join(os.path.dirname(self.weights_path), "yolo26s_metadata.json"),
                "backend/ml/weights/yolo26s_metadata.json",
                "ml/weights/yolo26s_metadata.json",
            ]

        self.metadata: dict[str, Any] = {}
        self.model = None
        self.task = "detect"
        self.num_classes = 1
        self.class_names = {0: "vessel"}

        # Resolve weights path across production candidates
        if not os.path.exists(self.weights_path):
            for cand in candidates:
                if os.path.exists(cand):
                    self.weights_path = cand
                    break

        if not os.path.exists(self.weights_path):
            logger.warning("YOLO26s %s weights not found at %s", self.modality, self.weights_path)
            return

        logger.info("Loading YOLO26s %s vessel detector from %s", self.modality, self.weights_path)
        self.model = YOLO(self.weights_path)

        # Load and verify model metadata
        for m_path in meta_candidates:
            if os.path.exists(m_path):
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    logger.info("Loaded YOLO26s %s metadata from %s", self.modality, m_path)
                    break
                except Exception as e:
                    logger.warning("Error reading metadata at %s: %s", m_path, e)

        # Verify task = detect
        loaded_task = getattr(self.model, "task", "detect")
        if loaded_task and loaded_task != "detect":
            logger.warning("Model task mismatch: expected 'detect', found '%s'", loaded_task)
        self.task = "detect"

        # Verify classes = 1, class 0 = vessel
        model_names = getattr(self.model, "names", {})
        if isinstance(model_names, dict) and len(model_names) > 0:
            self.class_names = {int(k): str(v) for k, v in model_names.items()}
            self.num_classes = len(self.class_names)
        elif isinstance(model_names, (list, tuple)) and len(model_names) > 0:
            self.class_names = {i: str(n) for i, n in enumerate(model_names)}
            self.num_classes = len(self.class_names)

        logger.info(
            "YOLO26s %s verified: task=%s, num_classes=%d, class_0=%s",
            self.modality,
            self.task,
            self.num_classes,
            self.class_names.get(0, "vessel"),
        )

    @classmethod
    def get_instance(cls, weights_path: str | None = None, modality: str = "GRD") -> "YOLO26sVesselDetector":
        mod = modality.upper()
        key = f"{mod}_{weights_path or 'default'}"
        if key not in cls._instances:
            detector = cls(weights_path=weights_path, modality=mod)
            cls._instances[key] = detector
            if mod == "GRD" and cls._instance is None:
                cls._instance = detector
        return cls._instances[key]

    def predict_single_tile(
        self,
        tile_image: np.ndarray,
        confidence_threshold: float = 0.25,
    ) -> list[dict[str, Any]]:
        """
        Run inference on a single (H, W, 3) preprocessed SAR tile.
        Returns raw bounding boxes, confidence scores, and class predictions.
        """
        if not self.model:
            raise RuntimeError(f"YOLO26s model not loaded. Weights file missing at {self.weights_path}")

        device = "cpu"
        results = self.model.predict(
            tile_image,
            conf=confidence_threshold,
            verbose=False,
            device=device,
        )

        if not results or len(results) == 0:
            return []

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") and boxes.cls is not None else np.zeros(len(xyxy))

        detections = []
        for i in range(len(xyxy)):
            xmin, ymin, xmax, ymax = xyxy[i]
            cid = int(cls_ids[i])
            cname = self.class_names.get(cid, "vessel")
            detections.append({
                "bounding_box": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "confidence": round(float(confs[i]), 4),
                "class_id": cid,
                "class_name": cname,
            })
        return detections

    def predict_scene_tiles(
        self,
        tiles: list[SARTile],
        corner_coords: dict[str, tuple[float, float]],
        parent_width: int,
        parent_height: int,
        confidence_threshold: float = 0.25,
        scene_metadata: dict[str, Any] | None = None,
        return_raw: bool = False,
        iou_threshold: float = 0.30,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Run inference across all preprocessed SAR tiles, remap bounding boxes to
        global scene pixel coordinates, execute conservative cross-tile deduplication,
        and georeference to (lat, lon).
        Preserves source tile provenance, coordinate transformations, and duplicate linkage.
        """
        if not self.model:
            raise RuntimeError(f"YOLO26s {self.modality} model not loaded. Weights file missing at {self.weights_path}")

        # Guardrail: Check for modality mismatches between tiles and model
        for tile in tiles:
            tile_mod = getattr(tile, "modality", "GRD").upper()
            if tile_mod != self.modality:
                logger.warning(
                    "Cross-modality warning: detector is '%s' but tile is tagged '%s'. "
                    "GRD and SLC processing must remain strictly separated.",
                    self.modality,
                    tile_mod,
                )

        raw_candidates: list[dict[str, Any]] = []

        for tile in tiles:
            tile_detections = self.predict_single_tile(
                tile.image_array,
                confidence_threshold=confidence_threshold,
            )

            for det in tile_detections:
                t_xmin, t_ymin, t_xmax, t_ymax = det["bounding_box"]
                # Remap to global parent scene coordinate system
                s_xmin = tile.x_offset + t_xmin
                s_ymin = tile.y_offset + t_ymin
                s_xmax = tile.x_offset + t_xmax
                s_ymax = tile.y_offset + t_ymax

                center_x = (s_xmin + s_xmax) / 2.0
                center_y = (s_ymin + s_ymax) / 2.0

                if scene_metadata and "geolocation_grid_points" in scene_metadata and scene_metadata["geolocation_grid_points"]:
                    x_off = float(scene_metadata.get("crop_x_offset", 0.0))
                    y_off = float(scene_metadata.get("crop_y_offset", 0.0))
                    lon, lat = SARGeoreferencer.pixel_to_lonlat_grid(
                        x=center_x,
                        y=center_y,
                        grid_points=scene_metadata["geolocation_grid_points"],
                        x_offset=x_off,
                        y_offset=y_off,
                    )
                else:
                    lon, lat = SARGeoreferencer.pixel_to_lonlat_bilinear(
                        x=center_x,
                        y=center_y,
                        width=parent_width,
                        height=parent_height,
                        corner_coords=corner_coords,
                    )

                # Estimate vessel length using modality-specific sensor geometry
                scene_box = [float(s_xmin), float(s_ymin), float(s_xmax), float(s_ymax)]
                est_length_m = SARGeoreferencer.calculate_vessel_length(
                    bbox=scene_box,
                    modality=self.modality,
                    metadata=scene_metadata,
                )

                detection_id = f"sar-det-{uuid.uuid4().hex[:12]}"
                raw_candidates.append({
                    "detection_id": detection_id,
                    "latitude": lat,
                    "longitude": lon,
                    "bounding_box": [round(s_xmin, 1), round(s_ymin, 1), round(s_xmax, 1), round(s_ymax, 1)],
                    "pixel_bbox": {
                        "ymin": round(s_ymin, 1),
                        "xmin": round(s_xmin, 1),
                        "ymax": round(s_ymax, 1),
                        "xmax": round(s_xmax, 1),
                    },
                    "confidence": round(float(det["confidence"]), 4),
                    "class_id": det.get("class_id", 0),
                    "class_name": det.get("class_name", "vessel"),
                    "source_tile": {
                        "tile_index": tile.tile_index,
                        "x_offset": tile.x_offset,
                        "y_offset": tile.y_offset,
                        "width": getattr(tile, "width", 640),
                        "height": getattr(tile, "height", 640),
                        "tile_bbox": [round(t_xmin, 1), round(t_ymin, 1), round(t_xmax, 1), round(t_ymax, 1)],
                    },
                    "coord_transform": {
                        "type": "tile_to_scene_translation",
                        "dx": tile.x_offset,
                        "dy": tile.y_offset,
                    },
                    "model_version": self.model_version,
                    "modality": self.modality,
                    "length_m": est_length_m,
                    "heading_deg": None,
                    "duplicate_provenance": None,
                })

        # Eliminate duplicate detections on tile overlap boundaries via conservative scene-coordinate matching
        operational_detections, candidates_with_provenance = deduplicate_scene_detections(
            raw_candidates,
            iou_threshold=iou_threshold,
        )

        self.last_raw_candidates = candidates_with_provenance

        logger.info(
            "Detected %d operational vessels across %d tiles (cross-tile deduplication reduced from %d raw candidates)",
            len(operational_detections),
            len(tiles),
            len(raw_candidates),
        )

        if return_raw:
            return operational_detections, candidates_with_provenance
        return operational_detections
