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
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def robust_sar_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Robust 2nd to 99.5th percentile SAR contrast stretch to uint8 [0, 255].
    Vessels appear as bright distinct radar scatterers against dark sea clutter.
    """
    arr_f = arr.astype(np.float32)
    p2 = float(np.percentile(arr_f, 2.0))
    p99_5 = float(np.percentile(arr_f, 99.5))
    if p99_5 <= p2:
        p99_5 = p2 + 1e-3
    norm = np.clip((arr_f - p2) / (p99_5 - p2), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def parse_xml_labels_standard(xml_content: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse vessel bounding boxes from OpenSAR XML label content.
    Returns list of standard YOLO detection boxes: (class_id, xc, yc, w, h) in [0, 1].
    Annotations are referenced to the 512x512 grid.
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

            x_min = min(left, right)
            x_max = max(left, right)
            y_min = min(top, bottom)
            y_max = max(top, bottom)

            if x_max <= x_min or y_max <= y_min:
                continue

            # Standard YOLO box normalization (512x512 grid)
            xc = max(0.001, min(0.999, (x_min + x_max) / (2.0 * 512.0)))
            yc = max(0.001, min(0.999, (y_min + y_max) / (2.0 * 512.0)))
            w = max(0.002, min(1.0, (x_max - x_min) / 512.0))
            h = max(0.002, min(1.0, (y_max - y_min) / 512.0))

            # Class 0: vessel
            boxes.append((0, xc, yc, w, h))

    except Exception as exc:
        logger.warning("Error parsing XML: %s", exc)

    return boxes


def rebuild_dataset(
    filtered_csv: str = "data/training/opensar/yolo_dataset_filtered.csv",
    output_root: str = "data/processed/opensar_filtered",
):
    df = pd.read_csv(filtered_csv)

    def to_base_id(fn):
        m = re.search(r'(DB_OPENSAR_)(?:VV|VH)_(VD_\d+)', fn)
        return f"{m.group(1)}{m.group(2)}" if m else fn

    df["base_id"] = df["file_name"].apply(to_base_id)

    target_bases = {
        "train": set(df[df["split"] == "train"]["base_id"]),
        "val": set(df[df["split"] == "val"]["base_id"]),
        "test": set(df[df["split"] == "test"]["base_id"]),
    }

    logger.info("Target patch counts: train=%d, val=%d, test=%d",
                len(target_bases["train"]), len(target_bases["val"]), len(target_bases["test"]))

    dctx = zstd.ZstdDecompressor()
    overall_stats = {}

    for split in ["train", "val", "test"]:
        labels_archive = f"data/training/opensar/{split}/{split}_labels.tar.zst"
        patches_archive = f"data/training/opensar/{split}/{split}_patches.tar.zst"

        images_dir = os.path.join(output_root, "images", split)
        labels_dir = os.path.join(output_root, "labels", split)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        target_set = target_bases[split]

        # 1. Parse XML labels
        label_boxes: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
        logger.info("[%s] Reading labels from %s...", split, labels_archive)
        with open(labels_archive, "rb") as f:
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|*") as tar:
                    for member in tar:
                        if member.name.endswith(".xml"):
                            base = os.path.splitext(os.path.basename(member.name))[0]
                            if base in target_set:
                                content = tar.extractfile(member).read().decode("utf-8")
                                boxes = parse_xml_labels_standard(content)
                                label_boxes[base] = boxes

        logger.info("[%s] Parsed labels for %d target patches", split, len(label_boxes))

        # 2. Extract and fuse GRD images
        logger.info("[%s] Extracting GRD patches from %s...", split, patches_archive)
        grd_data: Dict[str, Dict[str, np.ndarray]] = {}

        with open(patches_archive, "rb") as f:
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|*") as tar:
                    for member in tar:
                        if "_GRD_" not in member.name or not member.name.endswith((".tiff", ".tif")):
                            continue
                        fname = os.path.basename(member.name)
                        m = re.match(r"(DB_OPENSAR_VD_\d+)_GRD_(VV|VH)\.tiff?", fname, re.IGNORECASE)
                        if not m:
                            continue
                        base_id = m.group(1)
                        if base_id not in target_set:
                            continue

                        pol = m.group(2).upper()
                        file_bytes = tar.extractfile(member).read()
                        try:
                            im = Image.open(io.BytesIO(file_bytes))
                            arr = np.array(im)
                            if base_id not in grd_data:
                                grd_data[base_id] = {}
                            grd_data[base_id][pol] = arr
                        except Exception as e:
                            logger.warning("[%s] Failed to load %s: %s", split, fname, e)

        # 3. Save fused normalized RGB images and 5-column YOLO labels
        saved_imgs = 0
        saved_vessels = 0

        for base_id, pols in grd_data.items():
            if "VV" in pols and "VH" in pols:
                ch_r = robust_sar_normalize(pols["VV"])
                ch_g = robust_sar_normalize(pols["VH"])
                ch_b = ((ch_r.astype(np.float32) + ch_g.astype(np.float32)) / 2.0).astype(np.uint8)
                stacked = np.stack([ch_r, ch_g, ch_b], axis=-1)
            elif "VV" in pols:
                u = robust_sar_normalize(pols["VV"])
                stacked = np.stack([u, u, u], axis=-1)
            elif "VH" in pols:
                u = robust_sar_normalize(pols["VH"])
                stacked = np.stack([u, u, u], axis=-1)
            else:
                continue

            # Save image
            img_out_path = os.path.join(images_dir, f"{base_id}.jpg")
            pil_img = Image.fromarray(stacked)
            pil_img.save(img_out_path, quality=95)
            saved_imgs += 1

            # Save label file (standard 5 columns)
            lbl_out_path = os.path.join(labels_dir, f"{base_id}.txt")
            boxes = label_boxes.get(base_id, [])
            with open(lbl_out_path, "w") as f_lbl:
                for cid, xc, yc, w, h in boxes:
                    f_lbl.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    saved_vessels += 1

        overall_stats[split] = {"images": saved_imgs, "vessels": saved_vessels}
        logger.info("[%s] Saved %d images, %d vessels", split, saved_imgs, saved_vessels)

    # 4. Generate clean dataset.yaml for standard detection
    dataset_yaml = {
        "path": os.path.abspath(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "vessel",
        },
    }

    yaml_path = os.path.join(output_root, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    logger.info("Dataset rebuilding complete. Statistics: %s", overall_stats)
    logger.info("Saved clean detection dataset.yaml to %s", yaml_path)


if __name__ == "__main__":
    rebuild_dataset()
