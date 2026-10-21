import json
from pathlib import Path

report_path = Path("artifacts/sar_model_comparison/independent_validation_batch1/report.md")
manifest_path = Path("artifacts/sar_model_comparison/independent_validation_batch1/manifest.json")

with open(manifest_path) as f:
    manifest = json.load(f)

md = []
md.append("# Independent Real-World Sentinel-1 SLC Validation: Batch 1 Final Audit Report\n")
md.append(f"**Execution Timestamp:** `{manifest['execution_timestamp']}`  ")
md.append(f"**Model Version:** `{manifest['model_version']}`  ")
md.append(f"**Preprocessing:** `{manifest['pipeline_configuration']}`  ")
md.append(f"**Operating Point:** Confidence Threshold $\\tau = 0.25$, NMS IoU $0.50$  ")
md.append(f"**Geographic Validator:** `{manifest['geographic_context_source']}`\n")

md.append("## 1. Executive Summary & Scientific Method Protocol\n")
md.append("This independent validation evaluates the generalizability of the Sentinel-1 SLC vessel detection pipeline across three distinct real-world satellite acquisitions completely independent of the historical OpenSAR training and evaluation distribution:")
md.append("1. **Zagros Mountains, Iran**: Geographically independent inland hard-negative to evaluate clutter suppression under extreme topographic layover.")
md.append("2. **Bay of Biscay**: Deep open-ocean water (>250 km offshore) to evaluate false alarm density under wave clutter.")
md.append("3. **Singapore Strait / Jurong Port**: World's busiest littoral chokepoint to qualitatively evaluate dense vessel cluster resolution and coastal boundary preservation.\n")

md.append("### Crucial Terminology & Evaluation Principles")
md.append("> [!IMPORTANT]")
md.append("> **1. Unconfirmed Radar Contacts vs. Confirmed Vessels:**")
md.append("> The 190 candidates detected in the Singapore Strait scene are **unconfirmed model radar contacts**, NOT 190 confirmed vessels. Because this satellite acquisition lacks synchronized ground-truth AIS transponder logs or coincident sub-meter optical satellite imagery, zero candidates can be treated as verified ground truth. AIS-unmatched candidates must NOT be assumed to be confirmed illicit 'dark vessels'.")
md.append(">")
md.append("> **2. Scene-Specific Observation vs. General False-Alarm Rate:**")
md.append("> The observation of zero candidates in the Zagros Mountains and Bay of Biscay scenes establishes **zero observed detections in those particular satellite acquisitions and footprints ($0.00\\text{ contacts per } 1,000\\text{ km}^2$)**. It does NOT imply a universal zero false-alarm rate across all possible high-relief landforms, variable sea states, or severe storm regimes.")
md.append(">")
md.append("> **3. Inadmissibility of Precision, Recall, and mAP:**")
md.append("> No precision, recall, or mAP metrics can genuinely be calculated for unlabelled scenes. Presenting quantitative accuracy metrics without exhaustive ground-truth bounding box annotations would be scientifically invalid.\n")

md.append("### Coverage & Actual Crop Extent Audit")
md.append("In earlier prototyping experiments, arbitrary sub-crops (e.g. 2,048 × 1,400 px) were utilized. **In this Batch-1 execution, the full burst rasters were completely analyzed across their entire pixel extent**:")
md.append("- **Zagros (Burst 293020)**: $25,448 \\times 1,510$ pixels (38.43 Mpx, 135 tiles of $640 \\times 640$ with 64px overlap)")
md.append("- **Biscay (Burst 313866)**: $25,098 \\times 1,515$ pixels (38.02 Mpx, 132 tiles of $640 \\times 640$ with 64px overlap)")
md.append("- **Singapore (Burst 37586)**: $24,781 \\times 1,500$ pixels (37.17 Mpx, 129 tiles of $640 \\times 640$ with 64px overlap)\n")

md.append("### Footprint Area Calculation")
md.append("The CDSE OData `/Bursts` catalog publishes a bounding envelope polygon that includes outer beam margins (~1,800–1,930 km²). In this audit, the exact geographic footprint was re-calculated via spherical polygon integration over the corner coordinates from the Sentinel-1 annotation XML `geolocationGridPoint` records corresponding directly to the exact pixels processed:")
md.append("- **Zagros Footprint Area:** **1,724.8 km²** (CDSE envelope: 1,927.4 km²)")
md.append("- **Biscay Footprint Area:** **1,572.4 km²** (CDSE envelope: 1,801.6 km²)")
md.append("- **Singapore Footprint Area:** **1,729.3 km²** (CDSE envelope: 1,932.6 km²)\n")

md.append("### Summary Detection Matrix\n")
md.append("| Scene | Operational Regime | Analyzed Extent (px) | Analyzed GCP Area | Raw Candidates | Retained Contacts | Ambiguous Coastal | Terrestrial Clutter | Confirmed Vessels | Operational Density | Status |")
md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
md.append("| **Zagros Mountains, Iran** | Inland Hard-Negative | $25,448 \\times 1,510$ | **1,724.8 km²** | 0 | **0** | 0 | 0 | **0** | **0.00 / 1,000 km²** | `SUCCESS` |")
md.append("| **Bay of Biscay** | Open-Water Baseline | $25,098 \\times 1,515$ | **1,572.4 km²** | 0 | **0** | 0 | 0 | **0** | **0.00 / 1,000 km²** | `SUCCESS` |")
md.append("| **Singapore Strait** | Dense Coastal / Port | $24,781 \\times 1,500$ | **1,729.3 km²** | 190 | **190** | 125 | 0 | **0** | **109.87 / 1,000 km²** | `SUCCESS` |")

md.append("\n---\n")

md.append("## 2. Detailed Scene Audits\n")

# Zagros
md.append("### Scene 1: Zagros Mountains, Iran (Inland Hard-Negative)\n")
md.append("- **Product ID:** `S1D_IW_SLC__1SDV_20261019T023006_20261019T023034_005078_0099B8_46CE.SAFE`")
md.append("- **Burst ID:** `293020` (IW2) | Acquisition: `2026-10-19 02:30:22 UTC`")
md.append("- **Analyzed Pixel Extent:** Full Burst `25,448 × 1,510` (38,426,480 px, 135 tiles of 640×640)")
md.append("- **GCP Footprint Polygon:** `(52.41°E, 29.30°N) -> (51.48°E, 29.45°N) -> (51.44°E, 29.28°N) -> (52.38°E, 29.14°N)`")
md.append("- **Analyzed Footprint Area:** **1,724.8 km²**")
md.append("- **Ground Truth Protocol:** *Deterministic Zero Ground Truth (N=0)*\n")
md.append("#### Findings:")
md.append("- **Raw Neural Network Candidates:** 0")
md.append("- **Operational False Positives:** **0** (Observed Density: **0.00 / 1,000 km²** in this scene)")
md.append("- **Clutter Rejections:** 0 (zero candidates reached validator)")
md.append("> [!NOTE] Across all 135 tiles of severe topographic layover, ridge backscatter, and radar shadow, the YOLO26s SLC model with Config A percentile normalization produced zero raw false detections, confirming that hard-negative suppression generalizes to complex inland alpine relief in this scene.\n")

# Biscay
md.append("### Scene 2: Bay of Biscay (Open-Water Baseline)\n")
md.append("- **Product ID:** `S1D_IW_SLC__1SDV_20261019T182825_20261019T182852_005088_009A10_1BF2.SAFE`")
md.append("- **Burst ID:** `313866` (IW3) | Acquisition: `2026-10-19 18:28:42 UTC`")
md.append("- **Analyzed Pixel Extent:** Full Burst `25,098 × 1,515` (38,023,470 px, 132 tiles of 640×640)")
md.append("- **GCP Footprint Polygon:** `(-8.49°W, 45.67°N) -> (-7.41°W, 45.79°N) -> (-7.45°W, 45.96°N) -> (-8.53°W, 45.83°N)`")
md.append("- **Analyzed Footprint Area:** **1,572.4 km²**")
md.append("- **Ground Truth Protocol:** *Unannotated Open Ocean Baseline*\n")
md.append("#### Findings:")
md.append("- **Offshore Verification:** Verified abyssal plain (>2,500 m water depth, >250 km from French and Spanish coasts).")
md.append("- **Raw Neural Network Candidates:** 0")
md.append("- **Operational Contacts:** **0** (Observed Density: **0.00 / 1,000 km²** in this scene)")
md.append("> [!NOTE] Confirms that under moderate open-ocean conditions, Config A dual-pol percentile normalization yields zero wave-clutter false alarms at $\\tau = 0.25$ across the 1,572.4 km² analyzed in this pass.\n")

# Singapore
md.append("### Scene 3: Singapore Strait / Jurong Port (Dense Littoral & Anchorage)\n")
md.append("- **Product ID:** `S1D_IW_SLC__1SDV_20261010T224745_20261010T224812_004959_0095AF_BD52.SAFE`")
md.append("- **Burst ID:** `37586` (IW2) | Acquisition: `2026-10-10 22:47:46 UTC`")
md.append("- **Analyzed Pixel Extent:** Full Burst `24,781 × 1,500` (37,171,500 px, 129 tiles of 640×640)")
md.append("- **GCP Footprint Polygon:** `(104.24°E, 1.07°N) -> (103.43°E, 1.24°N) -> (103.40°E, 1.08°N) -> (104.20°E, 0.90°N)`")
md.append("- **Analyzed Footprint Area:** **1,729.3 km²**")
md.append("- **Ground Truth Protocol:** *Qualitative Diagnostic Review (Zero Confirmed Ground Truth)*\n")
md.append("#### Findings:")
md.append("- **Confirmed Vessels:** **0** (No synchronized ground-truth AIS or optical labels).")
md.append("- **Raw Model Candidates:** **190 unconfirmed radar contacts**")
md.append("- **Retained Operational Contacts:** **190**")
md.append("  - Fairway / Open Water: **65** (Contact density: **37.59 contacts per 1,000 km²**)")
md.append("  - Ambiguous Coastal (within 10.0 km buffer): **125**")
md.append("- **Terrestrial Clutter Rejections:** **0** (Zero false alarms on Singapore or Batam landmasses)")
md.append("- **Operational Contact Density:** **109.87 contacts per 1,000 km²** (190 / 1,729.3 × 1,000)")
md.append("- **Nearest-Neighbor Separation:** Mean = 962 m, Median = 616 m, Min = 77 m")
md.append("- **Potential Proximity / Duplicate Flags (<150m):** **8 pairs** (16 candidates flagged for manual review)")

md.append("\n### Visual Review Package & Annotation Workflow")
md.append("- **Interactive Review Gallery:** An easy-to-use HTML gallery featuring instant context/confidence filtering, search, and native chip zoom is saved at: [`3_singapore_strait/gallery.html`](3_singapore_strait/gallery.html).")
md.append("- **Candidate Index:** [`3_singapore_strait/candidate_review_index.md`](3_singapore_strait/candidate_review_index.md)")
md.append("- **Manual Annotation Template:** A standardized CSV ready for blind human evaluation with blank `review_label` (probable vessel, likely clutter, ambiguous, duplicate), `review_notes`, and `reviewer` fields is saved at: [`3_singapore_strait/manual_annotation_template.csv`](3_singapore_strait/manual_annotation_template.csv).")
md.append("- **Duplicate Audit Summary:** [`3_singapore_strait/duplicate_review_summary.json`](3_singapore_strait/duplicate_review_summary.json).\n")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"Updated {report_path}")

brain_report = Path("/Users/aaryamanmishra/.gemini/antigravity/brain/f1d9efea-975f-46e4-b03b-e94fbaa2764b/independent_validation_batch1_report.md")
with open(brain_report, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"Updated {brain_report}")
