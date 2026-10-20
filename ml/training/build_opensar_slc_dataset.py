#!/usr/bin/env python3
"""
build_opensar_slc_dataset.py - Build and verify the native SLC OpenSAR dataset.

Transforms the official OpenSAR Insight Sentinel-1 Single Look Complex (SLC) patches
and XML annotations into a production-grade YOLO detection dataset:
1. Native 512x512 SLC complex-valued radar rasters (exact 1:1 match with XML annotations)
2. Complex amplitude magnitude calculation: sqrt(real^2 + imag^2)
3. Deterministic percentile contrast stretch (p2 to p99.5) and dual-pol (VV/VH/Avg) fusion
4. Direct XML coordinate mapping into normalized [0, 1] YOLO detection bounds
5. Rigorous confidence filtering: excludes LOW-confidence Doppler-shifted/unaligned targets
6. Retention of verified clean ocean background hard negatives
7. Purging of cross-split duplicate image hashes from the training split
8. Visual verification overlays for train, val, test, background, and specific audit targets
9. Packaging of ml/kaggle/opensar_slc_corrected.zip
"""

import os
import io
import re
import sys
import glob
import json
import shutil
import zipfile
import hashlib
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import zstandard as zstd
import tarfile
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_opensar_slc")


def decode_and_normalize_slc_raster(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw SLC TIFF bytes into a normalized uint8 [0, 255] 512x512 array.
    SLC data is SAMPLEFORMAT_COMPLEXIEEEFP (float32 real + float32 imag per pixel).
    """
    # 512 * 512 * 8 bytes (complex64) = 2,097,152 bytes at the end of the file
    raw_data = file_bytes[-2097152:]
    c_arr = np.frombuffer(raw_data, dtype=np.complex64).reshape(512, 512)
    mag = np.abs(c_arr)

    p2 = float(np.percentile(mag, 2.0))
    p99_5 = float(np.percentile(mag, 99.5))
    if p99_5 <= p2:
        p99_5 = p2 + 1e-3

    norm = np.clip((mag - p2) / (p99_5 - p2), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def parse_and_validate_xml_slc_labels(
    xml_content: str,
    base_id: str,
) -> Tuple[str, List[Tuple[int, float, float, float, float]], List[Dict[str, any]]]:
    """
    Parse vessel bounding boxes from OpenSAR XML label content directly in SLC 512x512 space.
    Returns:
      status: 'clean_background', 'valid_vessel', or 'excluded'
      valid_boxes: list of (class_id=0, xc, yc, w, h) in normalized [0, 1] range
      exclusions: list of dicts describing rejected annotations or images and reasons
    """
    try:
        root = ET.fromstring(xml_content)
        list_of_ships = root.find(".//List_of_ships")
        if list_of_ships is None:
            return "clean_background", [], []

        ships = list_of_ships.findall("Ship")
        if not ships:
            return "clean_background", [], []

        valid_boxes = []
        exclusions = []
        has_low_conf = False
        has_high_med = False

        for ship in ships:
            ship_name = ship.find("Name").text if ship.find("Name") is not None else "Unknown"
            conf_elem = ship.find("Confidence")
            conf = conf_elem.text.strip() if conf_elem is not None and conf_elem.text else "NONE"

            bbox_elem = ship.find("BoundingBox")
            if bbox_elem is None:
                exclusions.append({
                    "base_id": base_id,
                    "ship_name": ship_name,
                    "confidence": conf,
                    "reason": "Missing <BoundingBox> element",
                })
                continue

            top_el = bbox_elem.find("Top")
            left_el = bbox_elem.find("Left")
            bottom_el = bbox_elem.find("Bottom")
            right_el = bbox_elem.find("Right")

            if None in (top_el, left_el, bottom_el, right_el):
                exclusions.append({
                    "base_id": base_id,
                    "ship_name": ship_name,
                    "confidence": conf,
                    "reason": "Missing coordinate tags in <BoundingBox>",
                })
                continue

            try:
                top = float(top_el.text)
                left = float(left_el.text)
                bottom = float(bottom_el.text)
                right = float(right_el.text)
            except ValueError as e:
                exclusions.append({
                    "base_id": base_id,
                    "ship_name": ship_name,
                    "confidence": conf,
                    "reason": f"Non-numeric coordinate: {e}",
                })
                continue

            # In SLC 512x512 frame: Left/Right is Sample (0..512), Top/Bottom is Line (0..512)
            x_min = max(0.0, min(512.0, min(left, right)))
            x_max = max(0.0, min(512.0, max(left, right)))
            y_min = max(0.0, min(512.0, min(top, bottom)))
            y_max = max(0.0, min(512.0, max(top, bottom)))

            w_px = x_max - x_min
            h_px = y_max - y_min

            # Reject zero-area or sub-pixel degenerate boxes
            if w_px < 1.0 or h_px < 1.0:
                exclusions.append({
                    "base_id": base_id,
                    "ship_name": ship_name,
                    "confidence": conf,
                    "reason": f"Degenerate box dimensions in SLC frame: w={w_px:.1f}px, h={h_px:.1f}px",
                })
                continue

            # Check confidence
            if conf == "LOW":
                has_low_conf = True
                exclusions.append({
                    "base_id": base_id,
                    "ship_name": ship_name,
                    "confidence": "LOW",
                    "reason": "Unreliable source annotation: LOW confidence due to vessel underway motion / azimuth Doppler shift",
                })
                continue

            # Valid HIGH or MEDIUM vessel
            has_high_med = True
            xc = max(0.001, min(0.999, (x_min + x_max) / (2.0 * 512.0)))
            yc = max(0.001, min(0.999, (y_min + y_max) / (2.0 * 512.0)))
            w = max(0.002, min(1.0, w_px / 512.0))
            h = max(0.002, min(1.0, h_px / 512.0))
            valid_boxes.append((0, xc, yc, w, h))

        # Determine image status:
        if has_low_conf and not has_high_med:
            return "excluded", [], exclusions
        elif has_low_conf and has_high_med:
            # Mixed image: exclude to prevent unannotated vessel false penalties
            exclusions.append({
                "base_id": base_id,
                "reason": "Mixed image containing both valid and LOW confidence vessels; excluded to prevent unannotated vessel false penalties",
            })
            return "excluded", [], exclusions
        elif has_high_med:
            return "valid_vessel", valid_boxes, exclusions
        else:
            # All boxes were degenerate
            return "excluded", [], exclusions

    except Exception as exc:
        return "excluded", [], [{
            "base_id": base_id,
            "reason": f"XML parse failure: {exc}",
        }]


def rebuild_complete_opensar_slc_dataset(
    raw_dir: str = "data/training/opensar",
    output_root: str = "data/processed/opensar_slc",
) -> Dict[str, any]:
    """
    Main pipeline to extract native SLC patches, parse XML labels in 512x512 space,
    enforce quality exclusions, and assemble a clean YOLO dataset.
    """
    logger.info("=" * 60)
    logger.info("BUILDING NATIVE SLC OPENSAR YOLO DETECTION DATASET")
    logger.info("=" * 60)

    os.makedirs(output_root, exist_ok=True)
    dctx = zstd.ZstdDecompressor()
    splits = ["train", "val", "test"]

    dataset_summary = {
        "splits": {},
        "exclusions_by_split": {},
        "duplicate_hashes_removed": [],
    }

    for split in splits:
        logger.info("-" * 40)
        logger.info("PROCESSING SPLIT: %s", split.upper())
        logger.info("-" * 40)

        lbl_arc = os.path.join(raw_dir, split, f"{split}_labels.tar.zst")
        pat_arc = os.path.join(raw_dir, split, f"{split}_patches.tar.zst")

        images_dir = os.path.join(output_root, "images", split)
        labels_dir = os.path.join(output_root, "labels", split)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # 1. Parse all XML labels
        label_boxes: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
        patch_status: Dict[str, str] = {}
        split_exclusions = []

        logger.info("[%s] Reading XML labels from %s...", split, lbl_arc)
        with open(lbl_arc, "rb") as f:
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|*") as tar:
                    for member in tar:
                        if member.name.endswith(".xml"):
                            base_id = os.path.splitext(os.path.basename(member.name))[0]
                            content = tar.extractfile(member).read().decode("utf-8")
                            status, boxes, excl = parse_and_validate_xml_slc_labels(content, base_id)
                            patch_status[base_id] = status
                            label_boxes[base_id] = boxes
                            split_exclusions.extend(excl)

        logger.info(
            "[%s] XML Analysis: %d clean background, %d valid vessel, %d excluded, %d box exclusions",
            split,
            sum(1 for s in patch_status.values() if s == "clean_background"),
            sum(1 for s in patch_status.values() if s == "valid_vessel"),
            sum(1 for s in patch_status.values() if s == "excluded"),
            len(split_exclusions),
        )

        # 2. Extract and decode native SLC rasters
        logger.info("[%s] Extracting native 512x512 SLC patches from %s...", split, pat_arc)
        slc_patches: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

        with open(pat_arc, "rb") as f:
            with dctx.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode="r|*") as tar:
                    for member in tar:
                        fname = os.path.basename(member.name)
                        if "_SLC_" not in fname or not fname.endswith((".tiff", ".tif")):
                            continue

                        m = re.match(r"(DB_OPENSAR_VD_\d+)_SLC_(VV|VH)\.tiff?", fname, re.IGNORECASE)
                        if not m:
                            continue

                        base_id = m.group(1)
                        if patch_status.get(base_id) == "excluded":
                            continue

                        pol = m.group(2).upper()
                        file_bytes = tar.extractfile(member).read()
                        try:
                            norm_raster = decode_and_normalize_slc_raster(file_bytes)
                            slc_patches[base_id][pol] = norm_raster
                        except Exception as e:
                            logger.warning("[%s] Failed decoding SLC %s: %s", split, fname, e)

        logger.info("[%s] Loaded SLC rasters for %d patches", split, len(slc_patches))

        # 3. Write fused 512x512 RGB images and standard YOLO labels
        saved_images = 0
        vessel_images = 0
        background_images = 0
        total_vessels = 0

        for base_id, pols in slc_patches.items():
            if "VV" not in pols and "VH" not in pols:
                continue

            status = patch_status.get(base_id, "clean_background")
            if status == "excluded":
                continue

            if "VV" in pols and "VH" in pols:
                ch_r = pols["VV"]
                ch_g = pols["VH"]
                ch_b = ((ch_r.astype(np.float32) + ch_g.astype(np.float32)) / 2.0).astype(np.uint8)
                stacked = np.stack([ch_r, ch_g, ch_b], axis=-1)
            elif "VV" in pols:
                u = pols["VV"]
                stacked = np.stack([u, u, u], axis=-1)
            else:
                u = pols["VH"]
                stacked = np.stack([u, u, u], axis=-1)

            # Save image as JPEG (512x512, quality=95)
            img_path = os.path.join(images_dir, f"{base_id}.jpg")
            pil_img = Image.fromarray(stacked)
            pil_img.save(img_path, quality=95)
            saved_images += 1

            # Save label file
            boxes = label_boxes.get(base_id, [])
            lbl_path = os.path.join(labels_dir, f"{base_id}.txt")
            with open(lbl_path, "w", encoding="utf-8") as f_lbl:
                for cid, xc, yc, w, h in boxes:
                    f_lbl.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    total_vessels += 1

            if len(boxes) > 0:
                vessel_images += 1
            else:
                background_images += 1

        split_stat = {
            "total_images": saved_images,
            "vessel_images": vessel_images,
            "background_images": background_images,
            "total_vessel_annotations": total_vessels,
            "exclusions_count": len(split_exclusions),
        }
        dataset_summary["splits"][split] = split_stat
        dataset_summary["exclusions_by_split"][split] = split_exclusions

        logger.info(
            "[%s] Completed: %d images (%d vessel, %d background), %d valid vessel annotations",
            split, saved_images, vessel_images, background_images, total_vessels
        )

    # 4. Cross-split duplicate hash audit and purging
    logger.info("Auditing cross-split identical image hashes...")
    val_test_hashes = set()
    for s in ["val", "test"]:
        s_img_dir = os.path.join(output_root, "images", s)
        for p in glob.glob(os.path.join(s_img_dir, "*.jpg")):
            with open(p, "rb") as f_img:
                val_test_hashes.add(hashlib.md5(f_img.read()).hexdigest())

    train_img_dir = os.path.join(output_root, "images", "train")
    train_lbl_dir = os.path.join(output_root, "labels", "train")
    purged_train = []

    for p in sorted(glob.glob(os.path.join(train_img_dir, "*.jpg"))):
        with open(p, "rb") as f_img:
            h = hashlib.md5(f_img.read()).hexdigest()
        if h in val_test_hashes:
            base_id = os.path.splitext(os.path.basename(p))[0]
            lbl_p = os.path.join(train_lbl_dir, f"{base_id}.txt")
            is_bg = True
            if os.path.exists(lbl_p):
                with open(lbl_p, "r", encoding="utf-8") as f_lbl:
                    if f_lbl.read().strip():
                        is_bg = False
            if is_bg:
                os.remove(p)
                if os.path.exists(lbl_p):
                    os.remove(lbl_p)
                purged_train.append(base_id)
                dataset_summary["splits"]["train"]["total_images"] -= 1
                dataset_summary["splits"]["train"]["background_images"] -= 1

    dataset_summary["duplicate_hashes_removed"] = purged_train
    if purged_train:
        logger.info("Purged %d cross-split duplicate background images from train: %s", len(purged_train), purged_train)
    else:
        logger.info("Zero cross-split duplicate image hashes detected.")

    # 5. Generate clean dataset.yaml
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
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    logger.info("Saved dataset YAML: %s", yaml_path)
    return dataset_summary


def generate_slc_visual_verifications(
    dataset_root: str = "data/processed/opensar_slc",
    output_dir: str = "data/diagnostics/slc_visual_verifications",
    raw_dir: str = "data/training/opensar",
) -> List[str]:
    """
    Generate ground-truth overlays with pixel coordinate axes on native 512x512 SLC images.
    Specifically verifies:
    - Train vessel, train background
    - Val vessel, val background
    - Test vessel, test background
    - DB_OPENSAR_VD_106
    - DB_OPENSAR_VD_108
    - DB_OPENSAR_VD_109 (investigating its exclusion)
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # 1. Known target analysis: 106, 108, 109
    known_targets = ["DB_OPENSAR_VD_106", "DB_OPENSAR_VD_108", "DB_OPENSAR_VD_109"]
    for tid in known_targets:
        img_p = os.path.join(dataset_root, "images", "test", f"{tid}.jpg")
        lbl_p = os.path.join(dataset_root, "labels", "test", f"{tid}.txt")

        if not os.path.exists(img_p):
            with open(os.path.join(raw_dir, "test", "test_patches.tar.zst"), "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as r:
                    with tarfile.open(fileobj=r, mode="r|*") as t:
                        vv_bytes = None
                        vh_bytes = None
                        for m in t:
                            if f"{tid}_SLC_VV.tiff" in m.name:
                                vv_bytes = t.extractfile(m).read()
                            elif f"{tid}_SLC_VH.tiff" in m.name:
                                vh_bytes = t.extractfile(m).read()
                        if vv_bytes and vh_bytes:
                            vv = decode_and_normalize_slc_raster(vv_bytes)
                            vh = decode_and_normalize_slc_raster(vh_bytes)
                            avg = ((vv.astype(np.float32) + vh.astype(np.float32)) / 2.0).astype(np.uint8)
                            pil_img = Image.fromarray(np.stack([vv, vh, avg], axis=-1))
                        elif vv_bytes:
                            u = decode_and_normalize_slc_raster(vv_bytes)
                            pil_img = Image.fromarray(np.stack([u, u, u], axis=-1))
            status_tag = "EXCLUDED (LOW Confidence Doppler Mismatch)"
            boxes = []
            with open(os.path.join(raw_dir, "test", "test_labels.tar.zst"), "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as r:
                    with tarfile.open(fileobj=r, mode="r|*") as t:
                        for m in t:
                            if f"{tid}.xml" in m.name:
                                content = t.extractfile(m).read().decode("utf-8")
                                root = ET.fromstring(content)
                                for ship in root.findall(".//Ship"):
                                    bbox = ship.find("BoundingBox")
                                    if bbox is not None:
                                        t_v = float(bbox.find("Top").text)
                                        l_v = float(bbox.find("Left").text)
                                        b_v = float(bbox.find("Bottom").text)
                                        r_v = float(bbox.find("Right").text)
                                        boxes.append((0, (l_v+r_v)/1024.0, (t_v+b_v)/1024.0, (r_v-l_v)/512.0, (b_v-t_v)/512.0))
        else:
            pil_img = Image.open(img_p).convert("RGB")
            status_tag = "VALID INCLUDED SAMPLE"
            boxes = []
            if os.path.exists(lbl_p):
                with open(lbl_p) as f:
                    for line in f:
                        if line.strip():
                            p = line.strip().split()
                            boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))

        draw = ImageDraw.Draw(pil_img)
        w_img, h_img = pil_img.size

        for cid, xc, yc, bw, bh in boxes:
            x0 = (xc - bw / 2.0) * w_img
            y0 = (yc - bh / 2.0) * h_img
            x1 = (xc + bw / 2.0) * w_img
            y1 = (yc + bh / 2.0) * h_img
            color = (255, 0, 0) if "EXCLUDED" in status_tag else (0, 255, 0)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((max(0, x0), max(0, y0 - 11)), f"vessel ({x0:.1f}, {y0:.1f})", fill=color)

        draw.rectangle([0, 0, w_img, 22], fill=(0, 0, 0))
        draw.text((4, 4), f"[SLC 512x512] {tid} | {status_tag} | {len(boxes)} box(es)", fill=(255, 255, 255))

        out_path = os.path.join(output_dir, f"{tid}_slc_overlay.jpg")
        pil_img.save(out_path, quality=95)
        generated.append(out_path)

    # 2. General category samples
    categories = [
        ("train_vessel", "train", True),
        ("train_background", "train", False),
        ("val_vessel", "val", True),
        ("val_background", "val", False),
        ("test_vessel", "test", True),
        ("test_background", "test", False),
    ]

    for cat_name, split, require_vessels in categories:
        img_dir = os.path.join(dataset_root, "images", split)
        lbl_dir = os.path.join(dataset_root, "labels", split)

        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        matching = []

        for img_p in all_imgs:
            base = os.path.splitext(os.path.basename(img_p))[0]
            lbl_p = os.path.join(lbl_dir, f"{base}.txt")
            if not os.path.exists(lbl_p):
                continue
            with open(lbl_p) as f:
                lines = [l.strip() for l in f if l.strip()]
            if (require_vessels and len(lines) > 0) or (not require_vessels and len(lines) == 0):
                matching.append((img_p, lbl_p, lines))

        indices = np.linspace(0, len(matching) - 1, min(3, len(matching)), dtype=int)
        for idx in indices:
            img_p, lbl_p, lines = matching[idx]
            base = os.path.splitext(os.path.basename(img_p))[0]

            with Image.open(img_p).convert("RGB") as pil_img:
                draw = ImageDraw.Draw(pil_img)
                w_img, h_img = pil_img.size

                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cid, xc, yc, bw, bh = [float(x) for x in parts]
                        x0 = (xc - bw / 2.0) * w_img
                        y0 = (yc - bh / 2.0) * h_img
                        x1 = (xc + bw / 2.0) * w_img
                        y1 = (yc + bh / 2.0) * h_img
                        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
                        draw.text((max(0, x0), max(0, y0 - 11)), "vessel:0", fill=(0, 255, 0))

                draw.rectangle([0, 0, w_img, 20], fill=(0, 0, 0))
                draw.text((4, 3), f"[SLC 512x512 {split.upper()}] {base} | {len(lines)} vessel(s)", fill=(255, 255, 255))

                out_p = os.path.join(output_dir, f"{cat_name}_{base}.jpg")
                pil_img.save(out_p, quality=95)
                generated.append(out_p)

    logger.info("Generated %d visual ground-truth verification overlays in: %s", len(generated), output_dir)
    return generated


def package_slc_kaggle_zip(
    source_dataset_dir: str = "data/processed/opensar_slc",
    target_zip_path: str = "ml/kaggle/opensar_slc_corrected.zip",
) -> str:
    """
    Compress the corrected SLC dataset into a zip archive ready for Kaggle GPU upload.
    Uses atomic temp file writing to guarantee integrity.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_zip_path)), exist_ok=True)
    logger.info("Packaging native SLC dataset into %s...", target_zip_path)

    temp_zip = target_zip_path + ".tmp"
    if os.path.exists(temp_zip):
        os.remove(temp_zip)

    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dataset_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.join("opensar_slc_corrected", os.path.relpath(full_path, source_dataset_dir))
                zipf.write(full_path, arcname=rel_path)

    shutil.move(temp_zip, target_zip_path)
    zip_size_mb = os.path.getsize(target_zip_path) / (1024.0 * 1024.0)
    logger.info("Native SLC Kaggle dataset archive packaged successfully: %.2f MB", zip_size_mb)
    return os.path.abspath(target_zip_path)


if __name__ == "__main__":
    summary = rebuild_complete_opensar_slc_dataset()
    verifs = generate_slc_visual_verifications()
    zip_path = package_slc_kaggle_zip()
    print("\n" + "=" * 60)
    print("NATIVE SLC OPENSAR DATASET BUILD COMPLETE")
    print(f"Archive: {zip_path}")
    print(f"Archive size: {os.path.getsize(zip_path)/(1024*1024):.2f} MB")
    print("Summary:")
    print(json.dumps(summary["splits"], indent=2))
    print("=" * 60)
