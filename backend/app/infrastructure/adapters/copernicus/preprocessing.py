import logging

import numpy as np

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
        modality: str = "GRD",
        padding: dict[str, int] | None = None,
        valid_width: int | None = None,
        valid_height: int | None = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ):
        self.tile_index = tile_index
        self.image_array = image_array  # uint8 (H, W, 3)
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width
        self.height = height
        self.parent_width = parent_width
        self.parent_height = parent_height
        self.modality = modality.upper()
        self.padding = padding or {"top": 0, "left": 0, "right": 0, "bottom": 0}
        self.valid_width = valid_width if valid_width is not None else width
        self.valid_height = valid_height if valid_height is not None else height
        self.scale_x = scale_x
        self.scale_y = scale_y


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
        vh_band: np.ndarray | None = None,
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

    def generate_tiles(self, scene_image: np.ndarray, modality: str = "GRD") -> list[SARTile]:
        """
        Divide a scene image (H, W, 3) into overlapping tiles of size tile_size x tile_size.
        Preserves image modality metadata on each tile.
        """
        h, w = scene_image.shape[:2]
        step = self.tile_size - self.tile_overlap

        if h <= self.tile_size and w <= self.tile_size:
            # Entire scene fits in one tile (or pad to tile_size)
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
            padded[:h, :w, :] = scene_image
            return [SARTile(0, padded, 0, 0, w, h, w, h, modality=modality)]

        tiles: list[SARTile] = []
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
                        modality=modality,
                    )
                )
                tile_idx += 1

                if x_end >= w:
                    break
            if y_end >= h:
                break

        logger.info(
            "Generated %d %s tiles of size %dx%d from scene %dx%d",
            len(tiles),
            modality.upper(),
            self.tile_size,
            self.tile_size,
            w,
            h,
        )
        return tiles

    def generate_native_chips(
        self,
        raster_image: np.ndarray,
        window_x_offset: int = 0,
        window_y_offset: int = 0,
        parent_width: int | None = None,
        parent_height: int | None = None,
        stride: int = 576,
        chip_size: int = 640,
        modality: str = "GRD",
    ) -> list[SARTile]:
        """
        Generate overlapping native-resolution chips of size (chip_size x chip_size)
        from a bounded raster window array (H, W, 3) using a configurable stride (default 576, 10% overlap).

        Boundary Handling:
        Partial boundary chips are padded with zeros to ensure constant input shape (640, 640, 3)
        while explicitly tracking padding offsets (pad_right, pad_bottom, valid_width, valid_height)
        so that subsequent detection processing can reject padding artifacts and unpad coordinates.
        """
        h, w = raster_image.shape[:2]
        p_w = parent_width or (window_x_offset + w)
        p_h = parent_height or (window_y_offset + h)

        tiles: list[SARTile] = []
        tile_idx = 0

        # Step through image using stride
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                win_w = min(chip_size, w - x)
                win_h = min(chip_size, h - y)

                sub_chip = raster_image[y : y + win_h, x : x + win_w, :]

                pad_right = chip_size - win_w
                pad_bottom = chip_size - win_h

                if pad_right > 0 or pad_bottom > 0:
                    padded_chip = np.zeros((chip_size, chip_size, 3), dtype=np.uint8)
                    padded_chip[:win_h, :win_w, :] = sub_chip
                    chip_array = padded_chip
                else:
                    chip_array = sub_chip

                tiles.append(
                    SARTile(
                        tile_index=tile_idx,
                        image_array=chip_array,
                        x_offset=window_x_offset + x,
                        y_offset=window_y_offset + y,
                        width=chip_size,
                        height=chip_size,
                        parent_width=p_w,
                        parent_height=p_h,
                        modality=modality,
                        padding={"top": 0, "left": 0, "right": pad_right, "bottom": pad_bottom},
                        valid_width=win_w,
                        valid_height=win_h,
                        scale_x=1.0,
                        scale_y=1.0,
                    )
                )
                tile_idx += 1

        logger.info(
            "Generated %d native %s chips of size %dx%d (stride=%d) from window %dx%d at offset (%d, %d)",
            len(tiles),
            modality.upper(),
            chip_size,
            chip_size,
            stride,
            w,
            h,
            window_x_offset,
            window_y_offset,
        )
        return tiles


class SLCSARPreprocessor(SARPreprocessor):
    """
    Dedicated Sentinel-1 Single Look Complex (SLC) preprocessor.
    Faithfully reproduces the training pipeline used for YOLO26s SLC:
    - Complex magnitude calculation: sqrt(real^2 + imag^2)
    - Deterministic percentile contrast stretch: 2.0% to 99.5%
    - Strict avoidance of logarithmic dB compression on SLC
    - Multi-channel (VV, VH, (VV+VH)/2) stacking
    """

    def __init__(self, tile_size: int = 640, tile_overlap: int = 64):
        super().__init__(tile_size=tile_size, tile_overlap=tile_overlap)

    @staticmethod
    def normalize_slc_band(
        band: np.ndarray,
        p_min: float = 2.0,
        p_max: float = 99.5,
    ) -> np.ndarray:
        """
        Normalize a raw SLC band (complex or float magnitude) into [0, 255] uint8.
        Computes sqrt(real^2 + imag^2) if complex, followed by [p2, p99.5] percentile clipping.
        """
        if np.iscomplexobj(band):
            # Complex amplitude magnitude sqrt(I^2 + Q^2)
            mag = np.abs(band)
        else:
            mag = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        p2 = float(np.percentile(mag, p_min))
        p99_5 = float(np.percentile(mag, p_max))
        if p99_5 <= p2:
            p99_5 = p2 + 1e-3

        norm = np.clip((mag - p2) / (p99_5 - p2), 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    @classmethod
    def decode_complex_slc_bytes(
        cls,
        file_bytes: bytes,
        width: int = 512,
        height: int = 512,
    ) -> np.ndarray:
        """
        Decode raw SLC binary slice (complex64: float32 I + float32 Q per pixel)
        into a normalized uint8 array matching OpenSAR SLC dataset decoder.
        """
        expected_bytes = width * height * 8
        raw_slice = file_bytes[-expected_bytes:]
        c_arr = np.frombuffer(raw_slice, dtype=np.complex64).reshape(height, width)
        return cls.normalize_slc_band(c_arr)

    def prepare_multichannel_slc(
        self,
        vv_band: np.ndarray,
        vh_band: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Prepare a 3-channel (H, W, 3) uint8 image from SLC VV and optional VH bands.
        Channel 0: VV (normalized complex magnitude)
        Channel 1: VH (normalized complex magnitude)
        Channel 2: Mean of VV and VH
        """
        vv_norm = self.normalize_slc_band(vv_band)

        if vh_band is not None and vh_band.shape == vv_band.shape:
            vh_norm = self.normalize_slc_band(vh_band)
            avg_norm = ((vv_norm.astype(np.float32) + vh_norm.astype(np.float32)) / 2.0).astype(np.uint8)
            stacked = np.stack([vv_norm, vh_norm, avg_norm], axis=-1)
        else:
            # Single-polarization fallback
            stacked = np.stack([vv_norm, vv_norm, vv_norm], axis=-1)

        return stacked

    def generate_tiles(self, scene_image: np.ndarray, modality: str = "SLC") -> list[SARTile]:
        """Generate tiles specifically tagged with SLC modality."""
        return super().generate_tiles(scene_image, modality="SLC")
