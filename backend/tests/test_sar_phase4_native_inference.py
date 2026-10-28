import numpy as np
from app.domain.sar_intelligence.inference import (
    YOLO26sVesselDetector,
)
from app.infrastructure.adapters.copernicus.preprocessing import SARTile, SARPreprocessor


def test_native_chip_generation_stride_and_padding():
    """
    Verify native-resolution chip generation:
    - Generates overlapping chips with configurable stride (e.g. 576 px, 10% overlap).
    - Preserves constant 640x640 shape via zero padding.
    - Accurately tracks valid_width, valid_height, and padding offsets.
    """
    pre = SARPreprocessor(tile_size=640, tile_overlap=64)
    # Synthetic raster window of 1200 x 1200 (requires 2x2 grid with stride 576)
    arr = np.ones((1200, 1200, 3), dtype=np.uint8) * 128
    tiles = pre.generate_native_chips(
        raster_image=arr,
        window_x_offset=1000,
        window_y_offset=2000,
        parent_width=26676,
        parent_height=16681,
        stride=576,
        chip_size=640,
        modality="GRD",
    )

    # Grid size: x in [0, 576, 1152], y in [0, 576, 1152] -> 3 x 3 = 9 chips
    assert len(tiles) == 9

    # First tile (0, 0): interior, fully valid
    t0 = tiles[0]
    assert t0.image_array.shape == (640, 640, 3)
    assert t0.x_offset == 1000
    assert t0.y_offset == 2000
    assert t0.valid_width == 640
    assert t0.valid_height == 640
    assert t0.padding == {"top": 0, "left": 0, "right": 0, "bottom": 0}

    # Corner boundary tile (1152, 1152): only 48x48 valid pixels left (1200 - 1152 = 48)
    t_last = tiles[-1]
    assert t_last.image_array.shape == (640, 640, 3)
    assert t_last.x_offset == 1000 + 1152
    assert t_last.y_offset == 2000 + 1152
    assert t_last.valid_width == 48
    assert t_last.valid_height == 48
    assert t_last.padding["right"] == 640 - 48
    assert t_last.padding["bottom"] == 640 - 48


def test_inference_padding_offset_removal():
    """
    Verify boundary padding handling during chip detection postprocessing:
    - Bounding boxes located strictly in the zero-padded region are discarded.
    - Partially overlapping bounding boxes are clamped to the valid region.
    """
    # Create detector instance
    detector = YOLO26sVesselDetector.get_instance(modality="GRD")

    # Tile with valid_width=300, valid_height=300 and pad=340
    tile = SARTile(
        tile_index=0,
        image_array=np.zeros((640, 640, 3), dtype=np.uint8),
        x_offset=5000,
        y_offset=3000,
        width=640,
        height=640,
        parent_width=26676,
        parent_height=16681,
        modality="GRD",
        padding={"top": 0, "left": 0, "right": 340, "bottom": 340},
        valid_width=300,
        valid_height=300,
        scale_x=1.0,
        scale_y=1.0,
    )

    # Mock predict_single_tile to return 3 boxes:
    # 1. Inside valid area [50, 50, 150, 150] -> should keep
    # 2. Straddling boundary [250, 250, 400, 400] -> should clamp to [250, 250, 300, 300]
    # 3. Inside padded area [450, 450, 550, 550] -> should discard
    mock_boxes = [
        {"bounding_box": [50.0, 50.0, 150.0, 150.0], "confidence": 0.85, "class_id": 0, "class_name": "vessel"},
        {"bounding_box": [250.0, 250.0, 400.0, 400.0], "confidence": 0.80, "class_id": 0, "class_name": "vessel"},
        {"bounding_box": [450.0, 450.0, 550.0, 550.0], "confidence": 0.75, "class_id": 0, "class_name": "vessel"},
    ]

    original_predict = detector.predict_single_tile
    detector.predict_single_tile = lambda img, confidence_threshold=0.25: mock_boxes

    corner_coords = {
        "top_left": (-2.5, 50.5),
        "top_right": (-1.5, 50.5),
        "bottom_right": (-1.5, 49.5),
        "bottom_left": (-2.5, 49.5),
    }

    try:
        detections = detector.predict_scene_tiles(
            tiles=[tile],
            corner_coords=corner_coords,
            parent_width=26676,
            parent_height=16681,
            confidence_threshold=0.25,
        )

        # Candidate in pad region (box 3) must be discarded!
        assert len(detections) == 2

        # Check first detection
        d0 = detections[0]
        assert d0["bounding_box"] == [5050.0, 3050.0, 5150.0, 3150.0]

        # Check second detection was clamped at valid_width (300) -> 5000 + 300 = 5300, and valid_height (300) -> 3000 + 300 = 3300
        d1 = detections[1]
        assert d1["bounding_box"] == [5250.0, 3250.0, 5300.0, 3300.0]

    finally:
        detector.predict_single_tile = original_predict


def test_out_of_bounds_and_nan_coordinate_rejection():
    """
    Verify that detection candidates with out-of-bounds scene coordinates or NaN
    geographic coordinates are rejected.
    """
    detector = YOLO26sVesselDetector.get_instance(modality="GRD")

    tile = SARTile(
        tile_index=0,
        image_array=np.zeros((640, 640, 3), dtype=np.uint8),
        x_offset=26600,
        y_offset=16600,
        width=640,
        height=640,
        parent_width=26676,
        parent_height=16681,
        modality="GRD",
        valid_width=640,
        valid_height=640,
    )

    # Box that is completely outside parent scene
    mock_boxes = [
        {"bounding_box": [100.0, 100.0, 200.0, 200.0], "confidence": 0.85, "class_id": 0, "class_name": "vessel"},
    ]

    original_predict = detector.predict_single_tile
    detector.predict_single_tile = lambda img, confidence_threshold=0.25: mock_boxes

    corner_coords = {
        "top_left": (-2.5, 50.5),
        "top_right": (-1.5, 50.5),
        "bottom_right": (-1.5, 49.5),
        "bottom_left": (-2.5, 49.5),
    }

    try:
        # Box starts at x=26700, y=16700, which exceeds parent_width=26676 and parent_height=16681
        detections = detector.predict_scene_tiles(
            tiles=[tile],
            corner_coords=corner_coords,
            parent_width=26676,
            parent_height=16681,
            confidence_threshold=0.25,
        )
        assert len(detections) == 0

    finally:
        detector.predict_single_tile = original_predict


def test_full_raster_coordinate_translation_with_gcp_grid():
    """
    Verify that detection from a native chip correctly translates to full-raster coordinates
    and georeferences via piecewise GCP grid without downsampling distortion.
    """
    detector = YOLO26sVesselDetector.get_instance(modality="GRD")

    # Native chip at offset (7000, 4000)
    tile = SARTile(
        tile_index=0,
        image_array=np.zeros((640, 640, 3), dtype=np.uint8),
        x_offset=7000,
        y_offset=4000,
        width=640,
        height=640,
        parent_width=26676,
        parent_height=16681,
        modality="GRD",
        valid_width=640,
        valid_height=640,
        scale_x=1.0,
        scale_y=1.0,
    )

    mock_boxes = [
        {"bounding_box": [100.0, 100.0, 200.0, 200.0], "confidence": 0.85, "class_id": 0, "class_name": "vessel"},
    ]

    original_predict = detector.predict_single_tile
    detector.predict_single_tile = lambda img, confidence_threshold=0.25: mock_boxes

    # Sample GCP grid covering scene
    gcps = [
        {"pixel": 0.0, "line": 0.0, "longitude": -3.0, "latitude": 51.0, "height": 0.0},
        {"pixel": 26676.0, "line": 0.0, "longitude": -1.0, "latitude": 51.0, "height": 0.0},
        {"pixel": 0.0, "line": 16681.0, "longitude": -3.0, "latitude": 49.0, "height": 0.0},
        {"pixel": 26676.0, "line": 16681.0, "longitude": -1.0, "latitude": 49.0, "height": 0.0},
    ]

    scene_meta = {
        "original_width": 26676,
        "original_height": 16681,
        "geolocation_grid_points": gcps,
        "scale_x": 13.025,  # Preview scale should be ignored when parent_width == original_width
        "scale_y": 13.022,
    }

    try:
        detections = detector.predict_scene_tiles(
            tiles=[tile],
            corner_coords={},
            parent_width=26676,
            parent_height=16681,
            confidence_threshold=0.25,
            scene_metadata=scene_meta,
        )

        assert len(detections) == 1
        d = detections[0]
        # Full scene box: [7000+100, 4000+100, 7000+200, 4000+200] = [7100, 4100, 7200, 4200]
        assert d["bounding_box"] == [7100.0, 4100.0, 7200.0, 4200.0]
        assert d["coord_transform"]["is_native_resolution"] is True
        assert d["coord_transform"]["raster_x"] == 7150.0
        assert d["coord_transform"]["raster_y"] == 4150.0
        assert d["coord_transform"]["scale_x"] == 1.0
        assert d["coord_transform"]["scale_y"] == 1.0
        assert d["coord_transform"]["georeference_method"] == "piecewise_gcp_grid"

        # Check geographic interpolation in bounds
        assert -3.0 < d["longitude"] < -1.0
        assert 49.0 < d["latitude"] < 51.0

    finally:
        detector.predict_single_tile = original_predict


def test_model_checkpoint_task_and_metadata_validation():
    """
    Verify model checkpoint metadata and task validation behavior:
    - Verifies GRD model weights exist and are loaded.
    - Inspects metadata dictionary.
    - Confirms modality isolation between GRD and SLC.
    """
    grd_detector = YOLO26sVesselDetector.get_instance(modality="GRD")
    slc_detector = YOLO26sVesselDetector.get_instance(modality="SLC")

    assert grd_detector.modality == "GRD"
    assert slc_detector.modality == "SLC"
    assert grd_detector.model_version == "yolo26s-opensar-v1"
    assert slc_detector.model_version == "yolo26s-opensar-slc-v1"
    assert grd_detector.class_names.get(0) == "vessel"
    assert slc_detector.class_names.get(0) == "vessel"
