import json
import csv
import math
import cv2
import numpy as np
from pathlib import Path
from searoute.utils import haversine

singapore_dir = Path("artifacts/sar_model_comparison/independent_validation_batch1/3_singapore_strait")
csv_path = singapore_dir / "detections.csv"

candidates = []
with open(csv_path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        candidates.append(row)

features = []

for c in candidates:
    cid = c["detection_id"]
    chip_path = singapore_dir / f"chips/chip_{cid}.jpg"
    chip = cv2.imread(str(chip_path)) # BGR: 0=Avg, 1=VH, 2=VV
    ch, cw = chip.shape[:2]
    
    # 1. Extract box inside chip
    # Ignore top 35px if height >= 50
    sub_y = 35 if ch >= 60 else 0
    sub = chip[sub_y:, :]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 100, 100), (95, 255, 255))
    pts = np.argwhere(mask > 0)
    
    if len(pts) > 0:
        ymin, xmin = pts.min(axis=0)
        ymax, xmax = pts.max(axis=0)
        ymin += sub_y
        ymax += sub_y
    else:
        # Fallback
        xmin = max(0, cw // 2 - 30)
        xmax = min(cw, cw // 2 + 30)
        ymin = max(sub_y, ch // 2 - 30)
        ymax = min(ch, ch // 2 + 30)
        
    bw = max(xmax - xmin, 2)
    bh = max(ymax - ymin, 2)
    aspect_ratio = max(bw, bh) / min(bw, bh)
    box_area = bw * bh
    
    # Target region (shrink by 2px to avoid the colored bounding box line itself)
    tx1 = min(xmin + 2, xmax - 1)
    ty1 = min(ymin + 2, ymax - 1)
    tx2 = max(xmax - 2, xmin + 1)
    ty2 = max(ymax - 2, ymin + 1)
    target_vv = chip[ty1:ty2, tx1:tx2, 2] # VV is channel 2 in BGR
    target_vh = chip[ty1:ty2, tx1:tx2, 1] # VH is channel 1 in BGR
    
    if target_vv.size == 0:
        target_vv = chip[ymin:ymax, xmin:xmax, 2]
        target_vh = chip[ymin:ymax, xmin:xmax, 1]

    # Clutter region: ring around box (10 to 30 px away)
    c_mask = np.zeros((ch, cw), dtype=np.uint8)
    rx1, ry1 = max(0, xmin - 30), max(sub_y, ymin - 30)
    rx2, ry2 = min(cw, xmax + 30), min(ch, ymax + 30)
    c_mask[ry1:ry2, rx1:rx2] = 1
    # Exclude target box plus 6px margin
    ex1, ey1 = max(0, xmin - 6), max(sub_y, ymin - 6)
    ex2, ey2 = min(cw, xmax + 6), min(ch, ymax + 6)
    c_mask[ey1:ey2, ex1:ex2] = 0
    
    clutter_vv = chip[:, :, 2][c_mask == 1]
    clutter_vh = chip[:, :, 1][c_mask == 1]
    
    t_mean_vv = float(np.mean(target_vv)) if target_vv.size > 0 else 1.0
    t_max_vv = float(np.max(target_vv)) if target_vv.size > 0 else 1.0
    c_mean_vv = float(np.mean(clutter_vv)) if clutter_vv.size > 0 else 1.0
    c_std_vv = float(np.std(clutter_vv)) if clutter_vv.size > 0 else 1.0
    
    t_mean_vh = float(np.mean(target_vh)) if target_vh.size > 0 else 1.0
    c_mean_vh = float(np.mean(clutter_vh)) if clutter_vh.size > 0 else 1.0
    
    # Signal-to-clutter ratio (SCR) in dB
    scr_vv_db = 20.0 * math.log10(max(t_mean_vv, 1e-2) / max(c_mean_vv, 1e-2))
    scr_vh_db = 20.0 * math.log10(max(t_mean_vh, 1e-2) / max(c_mean_vh, 1e-2))
    peak_clutter_ratio = t_max_vv / max(c_mean_vv, 1e-2)
    pol_contrast_ratio = 20.0 * math.log10(max(t_mean_vv, 1e-2) / max(t_mean_vh, 1e-2))
    
    # Target shape & connected components
    thresh_val = max(c_mean_vv + 2.0 * c_std_vv, 30.0)
    bin_target = (target_vv > thresh_val).astype(np.uint8)
    num_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(bin_target)
    cc_count = max(0, num_cc - 1)
    active_pixels = int(np.sum(bin_target))
    fill_ratio = active_pixels / max(box_area, 1)
    clutter_homogeneity = c_std_vv / max(c_mean_vv, 1e-2)
    
    features.append({
        "detection_id": cid,
        "confidence": float(c["confidence"]),
        "latitude": float(c["latitude"]),
        "longitude": float(c["longitude"]),
        "length_m": float(c["length_m"]),
        "status": c["status"],
        "distance_to_coast_km": float(c["distance_to_coast_km"]) if c["distance_to_coast_km"] else None,
        "box_width_px": bw,
        "box_height_px": bh,
        "box_aspect_ratio": round(aspect_ratio, 2),
        "target_mean_vv": round(t_mean_vv, 1),
        "target_max_vv": round(t_max_vv, 1),
        "clutter_mean_vv": round(c_mean_vv, 1),
        "scr_vv_db": round(scr_vv_db, 2),
        "scr_vh_db": round(scr_vh_db, 2),
        "peak_clutter_ratio": round(peak_clutter_ratio, 2),
        "pol_contrast_ratio_db": round(pol_contrast_ratio, 2),
        "connected_components_count": cc_count,
        "fill_ratio": round(fill_ratio, 3),
        "clutter_homogeneity": round(clutter_homogeneity, 2),
    })

print(f"Extracted features for {len(features)} candidates.")
print("Sample feature record:")
print(json.dumps(features[0], indent=2))
