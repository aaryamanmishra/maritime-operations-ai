#!/usr/bin/env python3
"""
dataset_check.py - Robust verification of OpenSAR detection dataset for YOLO26s.

Verifies:
1. Directory structure across train, val, and test splits
2. Standard 5-column YOLO detection format (class_id xc yc w h)
3. Single-class vessel label integrity (class_id == 0)
4. Absence of keypoint/pose configuration
5. Non-blank pixel intensity distribution
6. Generates diagnostic ground-truth bounding box visualizations
7. Outputs an execution-ready dataset_run.yaml configured for the current environment
"""

import os
import sys
import glob
import argparse
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_check")

# Candidate search directories in priority order for extracted dataset
CANDIDATE_DATASET_DIRS = [
    # Kaggle input paths
    "/kaggle/input/opensar-slc-corrected/opensar_slc_corrected",
    "/kaggle/input/opensar-slc-corrected",
    "/kaggle/input/opensar_slc_corrected",
    "/kaggle/input/opensar-detection/opensar_slc_corrected",
    # Kaggle working paths
    "/kaggle/working/opensar_slc_corrected",
    "/kaggle/working/data/opensar_slc_corrected",
    # Current directory
    "opensar_slc_corrected",
    # Local repository paths
    "data/processed/opensar_slc",
    "../data/processed/opensar_slc",
    "../../data/processed/opensar_slc",
]

# Candidate search locations for dataset zip archive
CANDIDATE_ZIP_PATHS = [
    "/kaggle/input/opensar-slc-corrected/opensar_slc_corrected.zip",
    "/kaggle/input/opensar_slc_corrected/opensar_slc_corrected.zip",
    "/kaggle/working/opensar_slc_corrected.zip",
    "opensar_slc_corrected.zip",
    "ml/kaggle/opensar_slc_corrected.zip",
    "../ml/kaggle/opensar_slc_corrected.zip",
    "../../ml/kaggle/opensar_slc_corrected.zip",
]


def extract_dataset_zip(zip_path: str, target_dir: Optional[str] = None) -> str:
    """Extract opensar_slc_corrected.zip and return the unzipped dataset root path."""
    import zipfile

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Determine extract target directory
    if target_dir is None:
        if os.path.isdir("/kaggle/working") and os.access("/kaggle/working", os.W_OK):
            target_dir = "/kaggle/working"
        else:
            target_dir = "."

    logger.info("Extracting dataset archive %s to %s ...", zip_path, target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    expected_root = os.path.join(target_dir, "opensar_slc_corrected")
    if os.path.isdir(os.path.join(expected_root, "images", "train")):
        logger.info("Successfully extracted dataset to: %s", os.path.abspath(expected_root))
        return os.path.abspath(expected_root)

    if os.path.isdir(os.path.join(target_dir, "images", "train")):
        logger.info("Successfully extracted dataset to: %s", os.path.abspath(target_dir))
        return os.path.abspath(target_dir)

    raise RuntimeError(
        f"Extraction succeeded, but images/train was not found under {expected_root} or {target_dir}."
    )


def find_dataset_dir(explicit_dir: Optional[str] = None) -> str:
    """Locate or unpack the OpenSAR SLC dataset root directory."""
    # 1. If explicit path is provided, check if it is a directory or a zip file
    if explicit_dir:
        if os.path.isfile(explicit_dir) and explicit_dir.lower().endswith(".zip"):
            logger.info("Explicit path is a zip file. Extracting: %s", explicit_dir)
            return extract_dataset_zip(explicit_dir)

        if os.path.isdir(explicit_dir):
            if os.path.isdir(os.path.join(explicit_dir, "images", "train")):
                logger.info("Using explicitly provided dataset directory: %s", explicit_dir)
                return os.path.abspath(explicit_dir)
            if os.path.isdir(os.path.join(explicit_dir, "opensar_slc_corrected", "images", "train")):
                nested = os.path.join(explicit_dir, "opensar_slc_corrected")
                logger.info("Found dataset in subfolder: %s", nested)
                return os.path.abspath(nested)

    # 2. Check candidate directories for pre-extracted dataset
    for path in CANDIDATE_DATASET_DIRS:
        if os.path.isdir(path):
            if os.path.isdir(os.path.join(path, "images", "train")):
                logger.info("Found OpenSAR dataset at: %s", os.path.abspath(path))
                return os.path.abspath(path)
            if os.path.isdir(os.path.join(path, "opensar_slc_corrected", "images", "train")):
                nested = os.path.join(path, "opensar_slc_corrected")
                logger.info("Found OpenSAR dataset at: %s", os.path.abspath(nested))
                return os.path.abspath(nested)

    # 3. Glob search for images/train in /kaggle/input if running on Kaggle
    if os.path.isdir("/kaggle/input"):
        for sub in glob.glob("/kaggle/input/**/images/train", recursive=True):
            parent = os.path.dirname(os.path.dirname(sub))
            logger.info("Discovered OpenSAR dataset via Kaggle input scan: %s", parent)
            return os.path.abspath(parent)

    # 4. If directory was not found, check candidate ZIP archives and extract
    for zip_candidate in CANDIDATE_ZIP_PATHS:
        if os.path.isfile(zip_candidate):
            logger.info("Located unextracted dataset archive at: %s", zip_candidate)
            return extract_dataset_zip(zip_candidate)

    # 5. Glob search for opensar*slc*.zip in /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for sub_zip in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            if "opensar" in os.path.basename(sub_zip).lower():
                logger.info("Discovered dataset archive via Kaggle input scan: %s", sub_zip)
                return extract_dataset_zip(sub_zip)

    raise FileNotFoundError(
        "Could not locate the OpenSAR SLC dataset directory or archive. Looked in:\n  - "
        + "\n  - ".join(CANDIDATE_DATASET_DIRS)
        + "\nCandidate zip archives checked:\n  - "
        + "\n  - ".join(CANDIDATE_ZIP_PATHS)
        + "\n\nPlease supply --data-dir pointing to opensar_slc_corrected or opensar_slc_corrected.zip."
    )


def audit_split(dataset_dir: str, split_name: str) -> Dict[str, any]:
    """Audit images, labels, and bounding boxes for a single split."""
    img_dir = os.path.join(dataset_dir, "images", split_name)
    lbl_dir = os.path.join(dataset_dir, "labels", split_name)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing images directory for split '{split_name}': {img_dir}")
    if not os.path.isdir(lbl_dir):
        raise FileNotFoundError(f"Missing labels directory for split '{split_name}': {lbl_dir}")

    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    labels = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

    total_images = len(images)
    total_labels = len(labels)
    total_boxes = 0
    class_counter: Dict[int, int] = {}
    box_widths: List[float] = []
    box_heights: List[float] = []
    invalid_lines = 0

    for lbl_path in labels:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    logger.error(
                        "INVALID LABEL FORMAT in %s: expected 5 tokens (class_id xc yc w h), got %d tokens: %r",
                        lbl_path, len(parts), line.strip()
                    )
                    invalid_lines += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError as exc:
                    logger.error("Non-numeric label value in %s: %s", lbl_path, exc)
                    invalid_lines += 1
                    continue

                if cls_id != 0:
                    logger.error("UNEXPECTED CLASS ID in %s: got %d (expected 0 for vessel)", lbl_path, cls_id)
                    invalid_lines += 1

                class_counter[cls_id] = class_counter.get(cls_id, 0) + 1
                total_boxes += 1
                box_widths.append(w)
                box_heights.append(h)

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    logger.warning("Box coordinates out of [0, 1] bounds in %s: %s", lbl_path, parts)

    return {
        "split": split_name,
        "images": total_images,
        "labels": total_labels,
        "total_boxes": total_boxes,
        "class_counter": class_counter,
        "invalid_lines": invalid_lines,
        "min_w": float(np.min(box_widths)) if box_widths else 0.0,
        "max_w": float(np.max(box_widths)) if box_widths else 0.0,
        "median_w": float(np.median(box_widths)) if box_widths else 0.0,
        "min_h": float(np.min(box_heights)) if box_heights else 0.0,
        "max_h": float(np.max(box_heights)) if box_heights else 0.0,
        "median_h": float(np.median(box_heights)) if box_heights else 0.0,
    }


def verify_image_qualities(dataset_dir: str, num_samples: int = 5) -> None:
    """Verify that images are non-blank, properly dynamic-range stretched SAR rasters."""
    all_imgs = sorted(glob.glob(os.path.join(dataset_dir, "images", "train", "*.jpg")))
    if not all_imgs:
        logger.warning("No train images found to inspect pixel statistics.")
        return

    logger.info("Inspecting pixel intensity distributions across sample images...")
    indices = np.linspace(0, len(all_imgs) - 1, min(num_samples, len(all_imgs)), dtype=int)
    for idx in indices:
        img_p = all_imgs[idx]
        with Image.open(img_p) as pil_img:
            arr = np.array(pil_img)
            min_v = int(arr.min())
            max_v = int(arr.max())
            mean_v = float(arr.mean())
            std_v = float(arr.std())
            shape = arr.shape
            base = os.path.basename(img_p)

            # Check for catastrophic blank image bug (e.g. min=255, max=255)
            if std_v < 1e-2 or (min_v == 255 and max_v == 255):
                raise ValueError(
                    f"CRITICAL: Image {base} is completely blank/solid white! "
                    f"min={min_v}, max={max_v}, std={std_v:.4f}. Preprocessing is corrupted."
                )

            logger.info("  - %s: shape=%s, range=[%d, %d], mean=%.2f, std=%.2f (PASS)",
                        base, shape, min_v, max_v, mean_v, std_v)


def render_ground_truth_samples(dataset_dir: str, vis_dir: str, num_samples: int = 5) -> List[str]:
    """Draw ground-truth bounding boxes on sample images to verify alignment visually."""
    os.makedirs(vis_dir, exist_ok=True)
    all_imgs = sorted(glob.glob(os.path.join(dataset_dir, "images", "train", "*.jpg")))
    if not all_imgs:
        return []

    indices = np.linspace(0, len(all_imgs) - 1, min(num_samples, len(all_imgs)), dtype=int)
    saved_paths = []

    for idx in indices:
        img_p = all_imgs[idx]
        base_stem = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(dataset_dir, "labels", "train", f"{base_stem}.txt")
        if not os.path.exists(lbl_p):
            continue

        with Image.open(img_p).convert("RGB") as pil_img:
            w_img, h_img = pil_img.size
            draw = ImageDraw.Draw(pil_img)

            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1]) * w_img
                        yc = float(parts[2]) * h_img
                        bw = float(parts[3]) * w_img
                        bh = float(parts[4]) * h_img
                        x0 = xc - bw / 2.0
                        y0 = yc - bh / 2.0
                        x1 = xc + bw / 2.0
                        y1 = yc + bh / 2.0
                        # Draw green rectangle for vessel ground-truth
                        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
                        draw.text((x0, max(0, y0 - 12)), f"vessel:{cls_id}", fill=(0, 255, 0))

            out_path = os.path.join(vis_dir, f"gt_{base_stem}.jpg")
            pil_img.save(out_path)
            saved_paths.append(out_path)

    logger.info("Saved %d ground-truth diagnostic visualization(s) to: %s", len(saved_paths), vis_dir)
    return saved_paths


def generate_dataset_run_yaml(dataset_dir: str, output_yaml: str) -> str:
    """Generate a clean dataset.yaml pointed at dataset_dir with standard detection setup."""
    yaml_config = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "vessel",
        },
    }

    out_dir = os.path.dirname(os.path.abspath(output_yaml))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Generated clean execution YAML: %s", os.path.abspath(output_yaml))
    return os.path.abspath(output_yaml)


def run_dataset_check(
    data_dir: Optional[str] = None,
    output_yaml: str = "dataset_run.yaml",
    vis_dir: str = "ground_truth_samples",
) -> Tuple[bool, str, Dict[str, any]]:
    """Run full audit and generate verified dataset_run.yaml."""
    logger.info("=== STARTING OPENSAR DATASET AUDIT ===")
    dataset_dir = find_dataset_dir(data_dir)

    split_reports = {}
    total_invalid = 0

    for split in ["train", "val", "test"]:
        report = audit_split(dataset_dir, split)
        split_reports[split] = report
        total_invalid += report["invalid_lines"]

        logger.info(
            "Split '%s': %d images, %d label files, %d boxes, class distribution: %s",
            split,
            report["images"],
            report["labels"],
            report["total_boxes"],
            report["class_counter"],
        )
        logger.info(
            "  Box Width:  [min=%.4f, median=%.4f, max=%.4f]",
            report["min_w"], report["median_w"], report["max_w"],
        )
        logger.info(
            "  Box Height: [min=%.4f, median=%.4f, max=%.4f]",
            report["min_h"], report["median_h"], report["max_h"],
        )

    verify_image_qualities(dataset_dir)
    render_ground_truth_samples(dataset_dir, vis_dir)
    generated_yaml = generate_dataset_run_yaml(dataset_dir, output_yaml)

    if total_invalid > 0:
        logger.error("Dataset audit FAILED: found %d invalid label lines.", total_invalid)
        return False, generated_yaml, split_reports

    logger.info("=== OPENSAR DATASET AUDIT PASSED: ZERO INVALID LABELS, STANDARD 5-COL FORMAT ===")
    return True, generated_yaml, split_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit OpenSAR detection dataset for YOLO26s.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to opensar_slc_corrected dataset root or archive")
    parser.add_argument("--output-yaml", type=str, default="dataset_run.yaml", help="Path to save generated dataset YAML")
    parser.add_argument("--vis-dir", type=str, default="ground_truth_samples", help="Path to save ground truth samples")
    args = parser.parse_args()

    success, y_path, _ = run_dataset_check(args.data_dir, args.output_yaml, args.vis_dir)
    if not success:
        sys.exit(1)
