import os
import io
import re
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from PIL import Image
import zstandard as zstd
import tarfile
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_xml_labels(xml_content: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse vessel bounding boxes from OpenSAR XML label content.
    Returns list of normalized YOLO boxes: (x_center, y_center, w, h) in [0, 1].
    Patch size is 512 x 512.
    """
    boxes = []
    try:
        root = ET.fromstring(xml_content)
        list_of_ships = root.find(".//List_of_ships")
        if list_of_ships is None:
            return []

        for ship in list_of_ships.findall("Ship"):
            bbox_elem = ship.find("BoundingBox")
            if bbox_elem is None:
                continue

            top_el = bbox_elem.find("Top")
            left_el = bbox_elem.find("Left")
            bottom_el = bbox_elem.find("Bottom")
            right_el = bbox_elem.find("Right")

            if None in (top_el, left_el, bottom_el, right_el):
                continue

            top = float(top_el.text)
            left = float(left_el.text)
            bottom = float(bottom_el.text)
            right = float(right_el.text)

            # Ensure coordinates are properly ordered
            x_min = min(left, right)
            x_max = max(left, right)
            y_min = min(top, bottom)
            y_max = max(top, bottom)

            # Ignore invalid / zero area boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            w_px = x_max - x_min
            h_px = y_max - y_min
            xc_px = x_min + w_px / 2.0
            yc_px = y_min + h_px / 2.0

            # Normalize by patch dimension (512x512)
            xc = max(0.0, min(1.0, xc_px / 512.0))
            yc = max(0.0, min(1.0, yc_px / 512.0))
            w = max(0.001, min(1.0, w_px / 512.0))
            h = max(0.001, min(1.0, h_px / 512.0))

            # Centroid keypoint (kpt_shape: [1, 3])
            kpt_x = xc
            kpt_y = yc
            centroid = ship.find("Centroid_Position")
            if centroid is not None:
                cx_el = centroid.find("Scene_Sample")
                cy_el = centroid.find("Scene_Line")
                if cx_el is not None and cy_el is not None and cx_el.text and cy_el.text:
                    try:
                        kpt_x = max(0.0, min(1.0, float(cx_el.text) / 512.0))
                        kpt_y = max(0.0, min(1.0, float(cy_el.text) / 512.0))
                    except ValueError:
                        pass

            boxes.append((xc, yc, w, h, kpt_x, kpt_y, 2))

    except Exception as exc:
        logger.warning("Error parsing XML: %s", exc)

    return boxes


def normalize_band_to_uint8(arr: np.ndarray, min_db: float = -30.0, max_db: float = 0.0) -> np.ndarray:
    """
    Convert SAR amplitude/intensity to dB and scale to [0, 255] uint8.
    """
    clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.maximum(clean, 1e-6)
    db = 10.0 * np.log10(clean)
    clipped = np.clip(db, min_db, max_db)
    norm = (clipped - min_db) / (max_db - min_db)
    return (norm * 255.0).astype(np.uint8)


def extract_split_data(
    split: str,
    labels_archive: str,
    patches_archive: str,
    output_root: str,
    max_samples: Optional[int] = None,
) -> int:
    """
    Extract GRD VV and VH patches and XML labels from .tar.zst archives,
    fuse to 3-channel RGB images, and save YOLO labels.
    """
    images_dir = os.path.join(output_root, "images", split)
    labels_dir = os.path.join(output_root, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 1. Load all labels into memory: key = patch_id (e.g. "DB_OPENSAR_VD_2024")
    label_boxes: Dict[str, List[Tuple[float, float, float, float]]] = {}
    dctx = zstd.ZstdDecompressor()

    logger.info("Reading labels from %s...", labels_archive)
    with open(labels_archive, "rb") as f:
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                for member in tar:
                    if member.name.endswith(".xml"):
                        content = tar.extractfile(member).read().decode("utf-8")
                        # e.g. "labels/DB_OPENSAR_VD_2024.xml" -> "DB_OPENSAR_VD_2024"
                        base = os.path.splitext(os.path.basename(member.name))[0]
                        boxes = parse_xml_labels(content)
                        label_boxes[base] = boxes

    logger.info("Loaded labels for %d patches in %s split", len(label_boxes), split)

    # 2. Extract GRD images: group by patch base ID
    logger.info("Reading patches from %s...", patches_archive)
    grd_files: Dict[str, Dict[str, Image.Image]] = {}  # base_id -> {"VV": img, "VH": img}

    with open(patches_archive, "rb") as f:
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                for member in tar:
                    if "_GRD_" not in member.name or not member.name.endswith((".tiff", ".tif")):
                        continue

                    # e.g. "patches/DB_OPENSAR_VD_2024_GRD_VV.tiff"
                    fname = os.path.basename(member.name)
                    m = re.match(r"(DB_OPENSAR_VD_\d+)_GRD_(VV|VH)\.tiff?", fname, re.IGNORECASE)
                    if not m:
                        continue

                    base_id = m.group(1)
                    pol = m.group(2).upper()

                    # Only process if we have corresponding label
                    if base_id not in label_boxes:
                        continue

                    file_bytes = tar.extractfile(member).read()
                    try:
                        img = Image.open(io.BytesIO(file_bytes))
                        arr = np.array(img)
                        if base_id not in grd_files:
                            grd_files[base_id] = {}
                        grd_files[base_id][pol] = arr
                    except Exception as err:
                        logger.warning("Failed opening TIFF %s: %s", fname, err)

    logger.info("Extracted %d unique GRD patch sets for %s", len(grd_files), split)

    # 3. Fuse and write YOLO samples
    written = 0
    for base_id, pol_dict in grd_files.items():
        if "VV" not in pol_dict and "VH" not in pol_dict:
            continue

        if "VV" in pol_dict and "VH" in pol_dict:
            vv = normalize_band_to_uint8(pol_dict["VV"], min_db=-30.0, max_db=0.0)
            vh = normalize_band_to_uint8(pol_dict["VH"], min_db=-35.0, max_db=-5.0)
            avg = ((vv.astype(np.float32) + vh.astype(np.float32)) / 2.0).astype(np.uint8)
            stacked = np.stack([vv, vh, avg], axis=-1)
        elif "VV" in pol_dict:
            vv = normalize_band_to_uint8(pol_dict["VV"], min_db=-30.0, max_db=0.0)
            stacked = np.stack([vv, vv, vv], axis=-1)
        else:
            vh = normalize_band_to_uint8(pol_dict["VH"], min_db=-35.0, max_db=-5.0)
            stacked = np.stack([vh, vh, vh], axis=-1)

        # Save fused image as JPEG
        out_img_path = os.path.join(images_dir, f"{base_id}.jpg")
        pil_img = Image.fromarray(stacked)
        pil_img.save(out_img_path, quality=95)

        # Save labels in YOLO format: <class_id> <x_center> <y_center> <width> <height>
        boxes = label_boxes.get(base_id, [])
        out_lbl_path = os.path.join(labels_dir, f"{base_id}.txt")
        with open(out_lbl_path, "w") as f_lbl:
            for b in boxes:
                # b: (xc, yc, w, h, kpt_x, kpt_y, 2)
                f_lbl.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f} {b[5]:.6f} {b[6]}\n")

        written += 1
        if max_samples and written >= max_samples:
            break

    logger.info("Successfully created %d YOLO training pairs in %s", written, split)
    return written


def prepare_opensar_yolo_dataset(raw_dir: str, output_dir: str):
    """
    Main entrypoint to convert OpenSAR Insight GRD dataset to YOLO format.
    """
    os.makedirs(output_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    stats = {}

    for s in splits:
        lbl_arc = os.path.join(raw_dir, s, f"{s}_labels.tar.zst")
        pat_arc = os.path.join(raw_dir, s, f"{s}_patches.tar.zst")

        if not os.path.exists(lbl_arc) or not os.path.exists(pat_arc):
            logger.error("Missing archive for split %s: %s, %s", s, lbl_arc, pat_arc)
            continue

        count = extract_split_data(s, lbl_arc, pat_arc, output_dir)
        stats[s] = count

    # Write dataset.yaml
    dataset_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [1, 3],
        "names": {
            0: "vessel",
        },
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    logger.info("Dataset YAML written to %s", yaml_path)
    logger.info("Dataset statistics: %s", stats)


if __name__ == "__main__":
    prepare_opensar_yolo_dataset(
        raw_dir="data/training/opensar",
        output_dir="data/processed/opensar_yolo",
    )
