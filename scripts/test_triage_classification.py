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

n = len(candidates)

# 1. First pass: Pairwise spatial distances and duplicate image correlation
close_pairs = []
for i in range(n):
    c1 = candidates[i]
    id1 = c1["detection_id"]
    pos1 = (float(c1["longitude"]), float(c1["latitude"]))
    
    for j in range(i + 1, n):
        c2 = candidates[j]
        id2 = c2["detection_id"]
        pos2 = (float(c2["longitude"]), float(c2["latitude"]))
        d_m = haversine(pos1, pos2) * 1000.0
        
        if d_m < 150.0:
            # Check chip image similarity
            im1 = cv2.imread(str(singapore_dir / f"chips/chip_{id1}.jpg"))
            im2 = cv2.imread(str(singapore_dir / f"chips/chip_{id2}.jpg"))
            
            # Align shapes if edge chip
            min_h = min(im1.shape[0], im2.shape[0])
            min_w = min(im1.shape[1], im2.shape[1])
            diff = float(np.mean(cv2.absdiff(im1[:min_h, :min_w], im2[:min_h, :min_w])))
            
            is_tile_dup = (diff < 16.0)
            close_pairs.append({
                "cand1": id1,
                "cand2": id2,
                "dist_m": round(d_m, 1),
                "pixel_diff": round(diff, 2),
                "is_tile_duplicate": is_tile_dup,
                "conf1": float(c1["confidence"]),
                "conf2": float(c2["confidence"]),
            })

# Identify which candidates are redundant duplicates (the one with lower confidence in tile duplicate pair)
redundant_duplicates = set()
duplicate_links = {}

for p in close_pairs:
    if p["is_tile_duplicate"]:
        if p["conf1"] >= p["conf2"]:
            primary = p["cand1"]
            redundant = p["cand2"]
        else:
            primary = p["cand2"]
            redundant = p["cand1"]
        redundant_duplicates.add(redundant)
        duplicate_links[redundant] = {
            "primary_id": primary,
            "dist_m": p["dist_m"],
            "pixel_diff": p["pixel_diff"]
        }

print(f"Total close pairs (<150m): {len(close_pairs)}")
tile_dup_count = sum(1 for p in close_pairs if p["is_tile_duplicate"])
distinct_close_count = len(close_pairs) - tile_dup_count
print(f"  -> Tile Overlap Duplicates: {tile_dup_count} pairs ({len(redundant_duplicates)} redundant candidates)")
print(f"  -> Distinct Nearby Vessels: {distinct_close_count} pairs")

# 2. Second pass: Feature extraction and Triage Category Assignment
triage_counts = {"HIGHER_PLAUSIBILITY": 0, "NEEDS_REVIEW": 0, "LIKELY_CLUTTER_OR_DUPLICATE": 0}

classified = []
for c in candidates:
    cid = c["detection_id"]
    conf = float(c["confidence"])
    length = float(c["length_m"])
    status = c["status"]
    dist_coast = float(c["distance_to_coast_km"]) if c["distance_to_coast_km"] else 999.0
    
    chip = cv2.imread(str(singapore_dir / f"chips/chip_{cid}.jpg"))
    ch, cw = chip.shape[:2]
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
        xmin, xmax = max(0, cw // 2 - 30), min(cw, cw // 2 + 30)
        ymin, ymax = max(sub_y, ch // 2 - 30), min(ch, ch // 2 + 30)
        
    bw = int(max(xmax - xmin, 2))
    bh = int(max(ymax - ymin, 2))
    aspect_ratio = float(max(bw, bh) / min(bw, bh))
    box_area = int(bw * bh)
    
    tx1, ty1 = min(xmin + 2, xmax - 1), min(ymin + 2, ymax - 1)
    tx2, ty2 = max(xmax - 2, xmin + 1), max(ymax - 2, ymin + 1)
    target_vv = chip[ty1:ty2, tx1:tx2, 2]
    target_vh = chip[ty1:ty2, tx1:tx2, 1]
    
    if target_vv.size == 0:
        target_vv = chip[ymin:ymax, xmin:xmax, 2]
        target_vh = chip[ymin:ymax, xmin:xmax, 1]

    c_mask = np.zeros((ch, cw), dtype=np.uint8)
    rx1, ry1 = max(0, xmin - 30), max(sub_y, ymin - 30)
    rx2, ry2 = min(cw, xmax + 30), min(ch, ymax + 30)
    c_mask[ry1:ry2, rx1:rx2] = 1
    ex1, ey1 = max(0, xmin - 6), max(sub_y, ymin - 6)
    ex2, ey2 = min(cw, xmax + 6), min(ch, ymax + 6)
    c_mask[ey1:ey2, ex1:ex2] = 0
    
    clutter_vv = chip[:, :, 2][c_mask == 1]
    clutter_vh = chip[:, :, 1][c_mask == 1]
    
    t_mean_vv = float(np.mean(target_vv)) if target_vv.size > 0 else 1.0
    t_max_vv = float(np.max(target_vv)) if target_vv.size > 0 else 1.0
    c_mean_vv = float(np.mean(clutter_vv)) if clutter_vv.size > 0 else 1.0
    c_std_vv = float(np.std(clutter_vv)) if clutter_vv.size > 0 else 1.0
    
    scr_vv_db = float(20.0 * math.log10(max(t_mean_vv, 1e-2) / max(c_mean_vv, 1e-2)))
    peak_clutter_ratio = float(t_max_vv / max(c_mean_vv, 1e-2))
    
    # Connected components
    thresh_val = max(c_mean_vv + 2.0 * c_std_vv, 30.0)
    bin_target = (target_vv > thresh_val).astype(np.uint8)
    num_cc, _, _, _ = cv2.connectedComponentsWithStats(bin_target)
    cc_count = max(0, num_cc - 1)
    fill_ratio = float(np.sum(bin_target) / max(box_area, 1))
    
    # TRIAGE RULES
    cat = ""
    rationale = []
    
    if cid in redundant_duplicates:
        cat = "LIKELY_CLUTTER_OR_DUPLICATE"
        dup_info = duplicate_links[cid]
        rationale.append(f"Confirmed tile-overlap duplicate of higher-confidence candidate {dup_info['primary_id']} (dist={dup_info['dist_m']}m, pixel diff={dup_info['pixel_diff']}).")
    elif scr_vv_db < 4.5 and conf < 0.28:
        cat = "LIKELY_CLUTTER_OR_DUPLICATE"
        rationale.append(f"Low SCR ({scr_vv_db:.1f} dB) combined with low confidence ({conf:.3f}) and weak peak scattering ratio ({peak_clutter_ratio:.1f}).")
    elif aspect_ratio > 8.0 and bw < 10:
        cat = "LIKELY_CLUTTER_OR_DUPLICATE"
        rationale.append(f"Extreme aspect ratio ({aspect_ratio:.1f}) and narrow line geometry ({bw}px) characteristic of radar streak / tile-edge artifact.")
    elif (conf >= 0.35 and scr_vv_db >= 8.0 and 40.0 <= length <= 350.0 and 1.3 <= aspect_ratio <= 6.5) or \
         (conf >= 0.40 and scr_vv_db >= 6.5 and 40.0 <= length <= 380.0):
        cat = "HIGHER_PLAUSIBILITY"
        rationale.append(f"Strong vessel morphology (conf={conf:.3f}, SCR={scr_vv_db:.1f} dB, len={length:.0f}m, aspect ratio={aspect_ratio:.1f}, {cc_count} scattering centers).")
    else:
        cat = "NEEDS_REVIEW"
        reasons = []
        if dist_coast < 2.0:
            reasons.append(f"nearshore coastal buffer ({dist_coast:.1f} km from coast)")
        if conf < 0.35:
            reasons.append(f"moderate/low confidence ({conf:.3f})")
        if scr_vv_db < 8.0:
            reasons.append(f"moderate contrast (SCR={scr_vv_db:.1f} dB)")
        if length > 300.0 or length < 50.0:
            reasons.append(f"extreme length ({length:.0f}m)")
        if not reasons:
            reasons.append("intermediate morphology")
        rationale.append(f"Ambiguous features requiring inspection: {', '.join(reasons)}.")
        
    triage_counts[cat] += 1
    classified.append((cid, cat, rationale[0]))

print("Triage Distribution:")
for k, v in triage_counts.items():
    print(f"  {k}: {v} ({v/190*100:.1f}%)")
