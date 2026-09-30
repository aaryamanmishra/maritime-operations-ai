import math
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SARTile:
    def __init__(
        self,
        tile_index: int,
        image_array: np.ndarray,
        x_offset: int,
        y_offset: int,
        width: int,
        height: int,
        parent_width: int,
        parent_height: int,
    ):
        self.tile_index = tile_index
        self.image_array = image_array  # uint8 (H, W, 3)
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width
        self.height = height
        self.parent_width = parent_width
        self.parent_height = parent_height


class SARPreprocessor:
    """
    Deterministic SAR preprocessing and tiling pipeline:
    - Radiometric log-stretch / dB normalization
    - Dual-polarization (VV/VH) channel fusion to 3-channel input
    - Overlapping spatial tiling
    - Preservation of georeferencing offsets
    """

    def __init__(self, tile_size: int = 640, tile_overlap: int = 64):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    @staticmethod
    def normalize_sar_band(band: np.ndarray, min_db: float = -30.0, max_db: float = 0.0) -> np.ndarray:
        """
        Convert raw amplitude / intensity to calibrated decibel scale and stretch to [0, 255] uint8.
        Supports both raw digital numbers and normalized float rasters.
        """
        arr = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        # If input is raw DN or uncalibrated uint16/float with values > 2.0, use robust percentile stretch
        if arr.max() > 2.0:
            p2 = float(np.percentile(arr, 2.0))
            p99_5 = float(np.percentile(arr, 99.5))
            if p99_5 <= p2:
                p99_5 = p2 + 1e-3
            norm = np.clip((arr - p2) / (p99_5 - p2), 0.0, 1.0)
            return (norm * 255.0).astype(np.uint8)

        # Standard linear intensity in [0, 1] to decibels
        arr = np.maximum(arr, 1e-6)
        db = 10.0 * np.log10(arr)
        clipped = np.clip(db, min_db, max_db)
        norm = (clipped - min_db) / (max_db - min_db)
        return (norm * 255.0).astype(np.uint8)

    def prepare_multichannel_sar(
        self,
        vv_band: np.ndarray,
        vh_band: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Prepare a 3-channel (H, W, 3) uint8 image from VV and optional VH bands.
        Channel 0: VV (calibrated ocean surface)
        Channel 1: VH (vessel cross-polarization target contrast)
        Channel 2: Mean or ratio
        """
        vv_norm = self.normalize_sar_band(vv_band, min_db=-30.0, max_db=0.0)

        if vh_band is not None and vh_band.shape == vv_band.shape:
            vh_norm = self.normalize_sar_band(vh_band, min_db=-35.0, max_db=-5.0)
            avg_norm = ((vv_norm.astype(np.float32) + vh_norm.astype(np.float32)) / 2.0).astype(np.uint8)
            stacked = np.stack([vv_norm, vh_norm, avg_norm], axis=-1)
        else:
            # Single polarization fallback: replicate across 3 channels
            stacked = np.stack([vv_norm, vv_norm, vv_norm], axis=-1)

        return stacked

    def generate_tiles(self, scene_image: np.ndarray) -> List[SARTile]:
        """
        Divide a scene image (H, W, 3) into overlapping tiles of size tile_size x tile_size.
        """
        h, w = scene_image.shape[:2]
        step = self.tile_size - self.tile_overlap

        if h <= self.tile_size and w <= self.tile_size:
            # Entire scene fits in one tile (or pad to tile_size)
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
            padded[:h, :w, :] = scene_image
            return [SARTile(0, padded, 0, 0, w, h, w, h)]

        tiles: List[SARTile] = []
        tile_idx = 0

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Ensure we don't exceed boundary
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                
                # Shift start back if hitting end border
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)

                chip = scene_image[y_start:y_end, x_start:x_end, :]

                # If chip is smaller than tile_size, pad it
                if chip.shape[0] < self.tile_size or chip.shape[1] < self.tile_size:
                    padded_chip = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    padded_chip[:chip.shape[0], :chip.shape[1], :] = chip
                    chip = padded_chip

                tiles.append(
                    SARTile(
                        tile_index=tile_idx,
                        image_array=chip,
                        x_offset=x_start,
                        y_offset=y_start,
                        width=min(self.tile_size, w - x_start),
                        height=min(self.tile_size, h - y_start),
                        parent_width=w,
                        parent_height=h,
                    )
                )
                tile_idx += 1

                if x_end >= w:
                    break
            if y_end >= h:
                break

        logger.info("Generated %d tiles of size %dx%d from scene %dx%d", len(tiles), self.tile_size, self.tile_size, w, h)
        return tiles
