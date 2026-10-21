import json
import csv
from pathlib import Path
from searoute.utils import haversine

singapore_dir = Path("artifacts/sar_model_comparison/independent_validation_batch1/3_singapore_strait")
csv_path = singapore_dir / "detections.csv"

candidates = []
with open(csv_path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        candidates.append(row)

# 1. Compute pairwise distances and duplicate flags
# Flag candidates with distance < 150m as potential duplicates / cluster contacts
n = len(candidates)
nearest_info = {}
duplicate_pairs = []

for i in range(n):
    c1 = candidates[i]
    id1 = c1["detection_id"]
    pos1 = (float(c1["longitude"]), float(c1["latitude"]))
    
    min_dist = float("inf")
    best_id = None
    
    for j in range(n):
        if i == j:
            continue
        c2 = candidates[j]
        pos2 = (float(c2["longitude"]), float(c2["latitude"]))
        d_m = haversine(pos1, pos2) * 1000.0
        
        if d_m < min_dist:
            min_dist = d_m
            best_id = c2["detection_id"]
            
        if j > i and d_m < 150.0:
            duplicate_pairs.append({
                "cand1": id1,
                "cand2": c2["detection_id"],
                "dist_m": round(d_m, 1),
                "conf1": float(c1["confidence"]),
                "conf2": float(c2["confidence"]),
                "len1": float(c1["length_m"]),
                "len2": float(c2["length_m"])
            })
            
    nearest_info[id1] = {
        "nearest_id": best_id,
        "dist_m": round(min_dist, 1),
        "is_potential_duplicate": min_dist < 150.0
    }

print(f"Total candidates: {len(candidates)}")
print(f"Total pairs within 150m: {len(duplicate_pairs)}")
flagged_count = sum(1 for v in nearest_info.values() if v["is_potential_duplicate"])
print(f"Total candidates flagged with neighbor < 150m: {flagged_count}")

# 2. Generate manual_annotation_template.csv
annotation_csv_path = singapore_dir / "manual_annotation_template.csv"
with open(annotation_csv_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "detection_id",
        "confidence",
        "latitude",
        "longitude",
        "length_m",
        "status",
        "distance_to_coast_km",
        "potential_duplicate_flag",
        "nearest_neighbor_id",
        "distance_to_neighbor_m",
        "chip_path",
        "review_label",
        "review_notes",
        "reviewer"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    
    # Sort by confidence descending
    sorted_candidates = sorted(candidates, key=lambda x: float(x["confidence"]), reverse=True)
    
    for c in sorted_candidates:
        cid = c["detection_id"]
        n_data = nearest_info[cid]
        writer.writerow({
            "detection_id": cid,
            "confidence": f"{float(c['confidence']):.4f}",
            "latitude": c["latitude"],
            "longitude": c["longitude"],
            "length_m": c["length_m"],
            "status": c["status"],
            "distance_to_coast_km": c["distance_to_coast_km"],
            "potential_duplicate_flag": "TRUE" if n_data["is_potential_duplicate"] else "FALSE",
            "nearest_neighbor_id": n_data["nearest_id"],
            "distance_to_neighbor_m": f"{n_data['dist_m']:.1f}",
            "chip_path": f"chips/chip_{cid}.jpg",
            "review_label": "", # BLANK: allowed: probable vessel, likely clutter, ambiguous, duplicate
            "review_notes": "", # BLANK
            "reviewer": ""      # BLANK
        })

print(f"Saved annotation template to {annotation_csv_path}")

# 3. Generate HTML Visual Review Gallery
gallery_html_path = singapore_dir / "gallery.html"

# Grouping breakdown for UI stats
total_high = sum(1 for c in candidates if float(c["confidence"]) >= 0.40)
total_med = sum(1 for c in candidates if 0.30 <= float(c["confidence"]) < 0.40)
total_low = sum(1 for c in candidates if float(c["confidence"]) < 0.30)

total_water = sum(1 for c in candidates if c["status"] == "PLAUSIBLE_WATER")
total_coastal = sum(1 for c in candidates if c["status"] == "AMBIGUOUS_COASTAL")

# Build HTML with embedded styles and dynamic client-side filtering
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Singapore Strait SLC Radar Candidates (Burst 37586) - Visual Review Gallery</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #131b2e;
      --card-border: #1e293b;
      --card-hover: #2d3b55;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --cyan: #38bdf8;
      --gold: #f59e0b;
      --red: #ef4444;
      --purple: #a855f7;
      --green: #10b981;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; }}
    header {{ border-bottom: 1px solid var(--card-border); padding-bottom: 20px; margin-bottom: 24px; }}
    h1 {{ font-size: 24px; color: var(--cyan); margin-bottom: 8px; display: flex; align-items: center; gap: 12px; }}
    .badge-count {{ background: #0284c7; color: white; padding: 2px 8px; border-radius: 12px; font-size: 13px; }}
    .subtitle {{ color: var(--text-muted); font-size: 14px; line-height: 1.5; }}
    .protocol-callout {{ background: #1e293b; border-left: 4px solid var(--gold); padding: 12px 16px; margin-top: 16px; font-size: 13px; border-radius: 4px; }}
    
    .stats-bar {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .stat-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 12px; }}
    .stat-val {{ font-size: 20px; font-weight: bold; }}
    .stat-lbl {{ font-size: 12px; color: var(--text-muted); }}

    .controls {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; margin-bottom: 24px; display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }}
    .control-group {{ display: flex; align-items: center; gap: 8px; }}
    label {{ font-size: 12px; font-weight: bold; color: var(--text-muted); text-transform: uppercase; }}
    select, input {{ background: #0f172a; border: 1px solid var(--card-border); color: var(--text); padding: 6px 12px; border-radius: 6px; font-size: 13px; outline: none; }}
    select:focus, input:focus {{ border-color: var(--cyan); }}

    .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; overflow: hidden; transition: transform 0.15s, border-color 0.15s; position: relative; }}
    .card:hover {{ transform: translateY(-2px); border-color: var(--card-hover); }}
    .card.dup-flag {{ border-color: #f43f5e; }}

    .chip-wrap {{ width: 100%; height: 220px; background: #000; overflow: hidden; position: relative; display: flex; align-items: center; justify-content: center; }}
    .chip-img {{ width: 100%; height: 100%; object-fit: contain; cursor: pointer; transition: transform 0.2s; }}
    .chip-img:hover {{ transform: scale(1.05); }}

    .card-body {{ padding: 12px; font-size: 12px; line-height: 1.6; }}
    .cid {{ font-family: monospace; font-weight: bold; color: var(--cyan); font-size: 12px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
    
    .status-badge {{ font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; text-transform: uppercase; }}
    .status-water {{ background: #0284c7; color: white; }}
    .status-coastal {{ background: #b45309; color: white; }}
    .dup-badge {{ background: #e11d48; color: white; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}

    .meta-row {{ display: flex; justify-content: space-between; color: var(--text-muted); }}
    .meta-val {{ color: var(--text); font-weight: 500; }}

    .overlay-link {{ display: block; text-align: center; margin-top: 8px; padding: 4px; background: #1e293b; color: var(--cyan); text-decoration: none; border-radius: 4px; font-size: 11px; }}
    .overlay-link:hover {{ background: #334155; }}

    /* Modal */
    .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); align-items: center; justify-content: center; }}
    .modal-content {{ max-width: 90%; max-height: 90%; border-radius: 8px; border: 2px solid var(--cyan); }}
  </style>
</head>
<body>

<header>
  <h1>Singapore Strait (Burst 37586) Candidate Review Gallery <span class="badge-count">190 Candidates</span></h1>
  <div class="subtitle">
    Product: <code>S1D_IW_SLC__1SDV_20261010T224745_...</code> | Acquisition: <code>2026-10-10 22:47:46 UTC</code> | Mode: <code>Sentinel-1 IW2 SLC</code><br>
    Model: <code>YOLO26s SLC (conf &ge; 0.25)</code> | Preprocessing: <code>Config A Full-Burst (24,781 &times; 1,500 px, 1,729.3 km&sup2;)</code>
  </div>
  <div class="protocol-callout">
    <strong>Scientific Evaluation Protocol:</strong> These 190 candidates represent unconfirmed neural radar detections. Because this acquisition lacks synchronized ground-truth AIS or optical labels, <strong>none are confirmed vessels</strong>. AIS-unmatched candidates must NOT be assumed to be illicit "dark vessels". Use the accompanying <a href="manual_annotation_template.csv" style="color:var(--cyan);">manual_annotation_template.csv</a> to record independent human review.
  </div>
</header>

<div class="stats-bar">
  <div class="stat-card">
    <div class="stat-val" style="color:var(--text);">{len(candidates)}</div>
    <div class="stat-lbl">Total Unconfirmed Candidates</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:var(--cyan);">{total_water}</div>
    <div class="stat-lbl">Plausible Water (Fairway)</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:var(--gold);">{total_coastal}</div>
    <div class="stat-lbl">Ambiguous Coastal (&lt;10km Buffer)</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:var(--purple);">{total_high}</div>
    <div class="stat-lbl">High Conf (&ge; 0.40)</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:var(--green);">{total_med}</div>
    <div class="stat-lbl">Med Conf (0.30 &ndash; 0.40)</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:#f43f5e;">{flagged_count}</div>
    <div class="stat-lbl">Nearby Flags (&lt;150m Separation)</div>
  </div>
</div>

<div class="controls">
  <div class="control-group">
    <label>Geographic Context:</label>
    <select id="filter-context" onchange="filterCards()">
      <option value="ALL">All Contexts (190)</option>
      <option value="PLAUSIBLE_WATER">Plausible Water ({total_water})</option>
      <option value="AMBIGUOUS_COASTAL">Ambiguous Coastal ({total_coastal})</option>
    </select>
  </div>

  <div class="control-group">
    <label>Confidence Range:</label>
    <select id="filter-conf" onchange="filterCards()">
      <option value="ALL">All Confidence Scores (190)</option>
      <option value="HIGH">&ge; 0.40 High ({total_high})</option>
      <option value="MED">0.30 &ndash; 0.40 Medium ({total_med})</option>
      <option value="LOW">0.25 &ndash; 0.30 Low ({total_low})</option>
    </select>
  </div>

  <div class="control-group">
    <label>Duplicate Warning:</label>
    <select id="filter-dup" onchange="filterCards()">
      <option value="ALL">All Candidates</option>
      <option value="DUP">Flagged &lt; 150m Apart ({flagged_count})</option>
    </select>
  </div>

  <div class="control-group">
    <label>Search:</label>
    <input type="text" id="search-input" placeholder="Search ID or coordinate..." onkeyup="filterCards()">
  </div>

  <div style="margin-left:auto;">
    <a href="overlay.png" target="_blank" style="color:var(--cyan); font-size:13px; text-decoration:none; font-weight:bold;">View Full-Burst Detection Map (overlay.png) &rarr;</a>
  </div>
</div>

<div class="gallery-grid" id="card-grid">
"""

for c in sorted_candidates:
    cid = c["detection_id"]
    conf = float(c["confidence"])
    lat = float(c["latitude"])
    lon = float(c["longitude"])
    length = float(c["length_m"]) if c["length_m"] else 0.0
    status = c["status"]
    dist = f"{float(c['distance_to_coast_km']):.2f} km" if c["distance_to_coast_km"] else "N/A"
    
    n_data = nearest_info[cid]
    is_dup = n_data["is_potential_duplicate"]
    dist_n = n_data["dist_m"]
    
    # Range bucket
    if conf >= 0.40:
        crange = "HIGH"
    elif conf >= 0.30:
        crange = "MED"
    else:
        crange = "LOW"
        
    status_cls = "status-water" if status == "PLAUSIBLE_WATER" else "status-coastal"
    dup_cls = "dup-flag" if is_dup else ""
    
    chip_src = f"chips/chip_{cid}.jpg"
    
    html += f"""
  <div class="card {dup_cls}" data-context="{status}" data-conf="{crange}" data-dup="{'DUP' if is_dup else 'NO'}" data-search="{cid} {lat} {lon}">
    <div class="chip-wrap">
      <img src="{chip_src}" class="chip-img" alt="{cid}" onclick="openModal(this.src)">
    </div>
    <div class="card-body">
      <div class="cid">
        <span>{cid[:12]}...</span>
        <span class="status-badge {status_cls}">{status[:12]}</span>
      </div>
      <div class="meta-row"><span>Confidence:</span><span class="meta-val" style="color:var(--cyan);">{conf:.4f}</span></div>
      <div class="meta-row"><span>Coordinates:</span><span class="meta-val">{lat:.4f}&deg;N, {lon:.4f}&deg;E</span></div>
      <div class="meta-row"><span>Est Length:</span><span class="meta-val">{length:.1f} m</span></div>
      <div class="meta-row"><span>Coast Dist:</span><span class="meta-val">{dist}</span></div>
      <div class="meta-row"><span>Neighbor Dist:</span><span class="meta-val" style="color:{'#f43f5e' if is_dup else 'inherit'}; font-weight:{'bold' if is_dup else 'normal'};">{dist_n:.0f} m</span></div>
      {f'<div style="margin-top:4px;"><span class="dup-badge">FLAGGED: &lt;150m to {n_data["nearest_id"][:10]}</span></div>' if is_dup else ''}
      <a href="{chip_src}" target="_blank" class="overlay-link">Open Full Chip Cutout</a>
    </div>
  </div>
"""

html += """
</div>

<div id="image-modal" class="modal" onclick="this.style.display='none'">
  <img class="modal-content" id="modal-img">
</div>

<script>
function filterCards() {
  const ctx = document.getElementById('filter-context').value;
  const conf = document.getElementById('filter-conf').value;
  const dup = document.getElementById('filter-dup').value;
  const q = document.getElementById('search-input').value.toLowerCase().trim();

  const cards = document.querySelectorAll('.card');
  cards.forEach(card => {
    const cardCtx = card.getAttribute('data-context');
    const cardConf = card.getAttribute('data-conf');
    const cardDup = card.getAttribute('data-dup');
    const cardSearch = card.getAttribute('data-search').toLowerCase();

    let show = true;
    if (ctx !== 'ALL' && cardCtx !== ctx) show = false;
    if (conf !== 'ALL' && cardConf !== conf) show = false;
    if (dup !== 'ALL' && cardDup !== dup) show = false;
    if (q && !cardSearch.includes(q)) show = false;

    card.style.display = show ? 'block' : 'none';
  });
}

function openModal(src) {
  const modal = document.getElementById('image-modal');
  const img = document.getElementById('modal-img');
  img.src = src;
  modal.style.display = 'flex';
}
</script>

</body>
</html>
"""

with open(gallery_html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved HTML gallery to {gallery_html_path}")

# 4. Save duplicate audit summary JSON
dup_summary_path = singapore_dir / "duplicate_review_summary.json"
with open(dup_summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_candidates": len(candidates),
        "proximity_threshold_meters": 150.0,
        "flagged_candidates_count": flagged_count,
        "close_pairs_count": len(duplicate_pairs),
        "close_pairs": duplicate_pairs
    }, f, indent=2)
print(f"Saved duplicate review summary to {dup_summary_path}")
