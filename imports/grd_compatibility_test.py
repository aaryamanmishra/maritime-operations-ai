#!/usr/bin/env python3
"""
grd_compatibility_test.py - Audit compatibility of SLC-trained models on real Sentinel-1 GRD imagery.
Compares:
1. Production Phase 6 model (trained on Sentinel-1 GRD native chips from LS-SSDD)
2. Experimental OpenSAR YOLO11s Baseline (640x640 SLC)
3. Experimental OpenSAR YOLO11s Paper (512x512 SLC)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

def load_grd_chips(grd_dir: Path, chip_size: int = 640, stride: int = 576) -> List[Dict[str, Any]]:
    """
    Extract native resolution chips from real Sentinel-1 GRD cached scene.
    """
    vv_path = grd_dir / "native_win_6000_3000_vv.tiff"
    vh_path = grd_dir / "native_win_6000_3000_vh.tiff"
    if not vv_path.exists():
        vv_path = grd_dir / "measurement_vv.tiff"
        vh_path = grd_dir / "measurement_vh.tiff"
    
    if not vv_path.exists():
        raise FileNotFoundError(f"No GRD raster found in {grd_dir}")
        
    import tifffile
    vv = tifffile.imread(str(vv_path))
    vh = tifffile.imread(str(vh_path)) if vh_path.exists() else vv
    
    # Normalize VV and VH to uint8 [0, 255]
    def norm_band(b):
        p2, p99_5 = np.percentile(b, (2.0, 99.5))
        if p99_5 <= p2:
            p99_5 = p2 + 1e-3
        return (np.clip((b - p2) / (p99_5 - p2), 0.0, 1.0) * 255.0).astype(np.uint8)

    vv_norm = norm_band(vv)
    vh_norm = norm_band(vh)
    mean_norm = ((vv_norm.astype(np.float32) + vh_norm) / 2.0).astype(np.uint8)
    rgb = np.stack([vv_norm, vh_norm, mean_norm], axis=-1)
    
    h, w, _ = rgb.shape
    chips = []
    chip_idx = 0
    for y in range(0, max(1, h - chip_size + 1), stride):
        for x in range(0, max(1, w - chip_size + 1), stride):
            crop = rgb[y:y+chip_size, x:x+chip_size]
            if crop.shape[0] != chip_size or crop.shape[1] != chip_size:
                # pad
                padded = np.zeros((chip_size, chip_size, 3), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            chips.append({
                "chip_id": f"grd_chip_{chip_idx}",
                "x": x,
                "y": y,
                "image": crop,
            })
            chip_idx += 1
            if len(chips) >= 12: # Test on representative 12 chips
                break
        if len(chips) >= 12:
            break
            
    return chips

def test_checkpoint_on_grd(model_path: str, chips: List[Dict[str, Any]], imgsz: int = 640, conf_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Run qualitative inference on real GRD chips.
    """
    model = YOLO(model_path)
    confs = []
    boxes_count = 0
    high_conf_count = 0
    detections_by_chip = []
    
    t0 = time.time()
    for chip in chips:
        img = chip["image"]
        # If model expects 512, resize or letterbox
        pil_img = Image.fromarray(img)
        preds = model(pil_img, imgsz=imgsz, conf=conf_threshold, verbose=False)[0]
        boxes = preds.boxes
        chip_dets = []
        if boxes is not None and len(boxes) > 0:
            c = boxes.conf.cpu().numpy().tolist()
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            confs.extend(c)
            boxes_count += len(c)
            high_conf_count += sum(1 for conf in c if conf >= 0.25)
            for box, conf in zip(xyxy, c):
                chip_dets.append({"box": [round(v, 1) for v in box], "conf": round(conf, 4)})
        detections_by_chip.append({
            "chip_id": chip["chip_id"],
            "detections": chip_dets
        })
    latency_ms = round(((time.time() - t0) / len(chips)) * 1000, 2)
    
    return {
        "model_path": model_path,
        "imgsz": imgsz,
        "chips_evaluated": len(chips),
        "total_detections_at_005": boxes_count,
        "detections_conf_ge_025": high_conf_count,
        "max_confidence": round(float(np.max(confs)), 4) if confs else 0.0,
        "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
        "median_confidence": round(float(np.median(confs)), 4) if confs else 0.0,
        "latency_ms_per_chip": latency_ms,
        "detections_by_chip": detections_by_chip,
    }

def main():
    grd_dir = Path("maritime-operations-ai/backend/data/sar_cache/grd/S1C_IW_GRDH_1SDV_20261026T062251_010052_0140FD_7A8C_COG")
    chips_640 = load_grd_chips(grd_dir, chip_size=640, stride=576)
    chips_512 = load_grd_chips(grd_dir, chip_size=512, stride=460)
    
    results = {}
    
    # 1. Test Phase 6 Production Model (trained on GRD native chips)
    prod_path = "maritime-operations-ai/backend/ml/weights/yolo26s_sar_vessel.pt"
    if os.path.exists(prod_path):
        print(f"Testing Phase 6 Production Model: {prod_path} (imgsz=640)...")
        results["phase6_production_grd"] = test_checkpoint_on_grd(prod_path, chips_640, imgsz=640, conf_threshold=0.05)
        print(f"  Detections >= 0.25: {results['phase6_production_grd']['detections_conf_ge_025']}, Max conf: {results['phase6_production_grd']['max_confidence']}")

    # 2. Test Experimental Baseline (if available)
    baseline_paths = [
        "maritime-operations-ai/backend/ml/weights/experimental/yolo11s_opensar_baseline/best.pt",
        "maritime-operations-ai/imports/opensar_checkpoints/baseline/best.pt",
        "maritime-operations-ai/backend/ml/weights/yolo26s_slc_vessel.pt", # previous SLC model for reference
    ]
    for bp in baseline_paths:
        if os.path.exists(bp):
            name = f"baseline_{Path(bp).parent.name}"
            print(f"Testing Experimental Candidate: {bp} (imgsz=640)...")
            results[name] = test_checkpoint_on_grd(bp, chips_640, imgsz=640, conf_threshold=0.05)
            print(f"  Detections >= 0.25: {results[name]['detections_conf_ge_025']}, Max conf: {results[name]['max_confidence']}")
            break

    # 3. Test Experimental Paper (if available)
    paper_paths = [
        "maritime-operations-ai/backend/ml/weights/experimental/yolo11s_opensar_paper/best.pt",
        "maritime-operations-ai/imports/opensar_checkpoints/paper/best.pt",
    ]
    for pp in paper_paths:
        if os.path.exists(pp):
            name = f"paper_{Path(pp).parent.name}"
            print(f"Testing Experimental Paper: {pp} (imgsz=512)...")
            results[name] = test_checkpoint_on_grd(pp, chips_512, imgsz=512, conf_threshold=0.05)
            print(f"  Detections >= 0.25: {results[name]['detections_conf_ge_025']}, Max conf: {results[name]['max_confidence']}")
            break

    out_path = Path("maritime-operations-ai/backend/ml/weights/experimental/grd_compatibility_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"GRD compatibility results written to: {out_path}")

if __name__ == "__main__":
    main()
