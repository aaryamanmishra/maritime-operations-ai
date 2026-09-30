import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from ultralytics import YOLO

from app.core.config import Settings
from app.infrastructure.adapters.copernicus.georeference import SARGeoreferencer
from app.infrastructure.adapters.copernicus.preprocessing import SARTile

logger = logging.getLogger(__name__)


def non_max_suppression_boxes(
    boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Standard greedy Non-Maximum Suppression (NMS) over candidate vessel detections.
    Each box is a dict containing 'bbox' [xmin, ymin, xmax, ymax] and 'confidence'.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    keep: List[Dict[str, Any]] = []

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


class YOLO26sVesselDetector:
    """
    Single production inference boundary for YOLO26s SAR vessel detection.
    Strictly performs inference only — no automated retraining inside backend container.
    """

    _instance: Optional["YOLO26sVesselDetector"] = None

    def __init__(self, weights_path: Optional[str] = None):
        settings = Settings()
        self.weights_path = weights_path or settings.SAR_MODEL_WEIGHTS_PATH
        self.metadata: Dict[str, Any] = {}
        self.model = None
        self.model_version = "yolo26s-opensar-v1"
        self.task = "detect"
        self.num_classes = 1
        self.class_names = {0: "vessel"}

        # Resolve weights path across production candidates
        if not os.path.exists(self.weights_path):
            candidates = [
                "backend/ml/weights/yolo26s_sar_vessel.pt",
                "ml/weights/yolo26s_sar_vessel.pt",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    self.weights_path = cand
                    break

        if not os.path.exists(self.weights_path):
            logger.warning("YOLO26s production weights not found at %s", self.weights_path)
            return

        logger.info("Loading YOLO26s production vessel detector from %s", self.weights_path)
        self.model = YOLO(self.weights_path)

        # Load and verify model metadata
        meta_candidates = [
            os.path.join(os.path.dirname(self.weights_path), "yolo26s_metadata.json"),
            "backend/ml/weights/yolo26s_metadata.json",
            "ml/weights/yolo26s_metadata.json",
        ]
        for m_path in meta_candidates:
            if os.path.exists(m_path):
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    logger.info("Loaded YOLO26s metadata from %s", m_path)
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
            "YOLO26s verified: task=%s, num_classes=%d, class_0=%s",
            self.task,
            self.num_classes,
            self.class_names.get(0, "vessel"),
        )

    @classmethod
    def get_instance(cls, weights_path: Optional[str] = None) -> "YOLO26sVesselDetector":
        if cls._instance is None:
            cls._instance = cls(weights_path)
        return cls._instance

    def predict_single_tile(
        self,
        tile_image: np.ndarray,
        confidence_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
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
        tiles: List[SARTile],
        corner_coords: Dict[str, Tuple[float, float]],
        parent_width: int,
        parent_height: int,
        confidence_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Run inference across all preprocessed SAR tiles, remap bounding boxes to
        global scene pixel coordinates, execute spatial NMS, and georeference to (lat, lon).
        Preserves source tile and scene metadata.
        """
        if not self.model:
            raise RuntimeError(f"YOLO26s model not loaded. Weights file missing at {self.weights_path}")

        raw_candidates: List[Dict[str, Any]] = []

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

                raw_candidates.append({
                    "bbox": [float(s_xmin), float(s_ymin), float(s_xmax), float(s_ymax)],
                    "confidence": det["confidence"],
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "tile_index": tile.tile_index,
                    "tile_x_offset": tile.x_offset,
                    "tile_y_offset": tile.y_offset,
                })

        # Eliminate duplicate detections on tile overlap boundaries via NMS
        filtered_candidates = non_max_suppression_boxes(raw_candidates, iou_threshold=0.35)

        # Georeference filtered detections
        georeferenced: List[Dict[str, Any]] = []
        for cand in filtered_candidates:
            s_xmin, s_ymin, s_xmax, s_ymax = cand["bbox"]
            center_x = (s_xmin + s_xmax) / 2.0
            center_y = (s_ymin + s_ymax) / 2.0

            lon, lat = SARGeoreferencer.pixel_to_lonlat_bilinear(
                x=center_x,
                y=center_y,
                width=parent_width,
                height=parent_height,
                corner_coords=corner_coords,
            )

            # Approximate vessel length in meters (assuming ~10m ground range resolution for S1 GRDH)
            box_diag_px = ((s_xmax - s_xmin) ** 2 + (s_ymax - s_ymin) ** 2) ** 0.5
            est_length_m = round(box_diag_px * 10.0, 1)

            detection_id = f"sar-det-{uuid.uuid4().hex[:12]}"
            georeferenced.append({
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
                "confidence": round(cand["confidence"], 4),
                "class_id": cand.get("class_id", 0),
                "class_name": cand.get("class_name", "vessel"),
                "source_tile": {
                    "tile_index": cand.get("tile_index", 0),
                    "x_offset": cand.get("tile_x_offset", 0),
                    "y_offset": cand.get("tile_y_offset", 0),
                },
                "model_version": self.model_version,
                "length_m": est_length_m,
                "heading_deg": None,
            })

        logger.info(
            "Detected %d vessels across %d tiles (NMS reduced from %d candidates)",
            len(georeferenced),
            len(tiles),
            len(raw_candidates),
        )
        return georeferenced
