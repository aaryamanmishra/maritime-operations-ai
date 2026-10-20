# Maritime Operations AI — Kaggle GPU Training Suite

This directory contains the autonomous training packages for the two ML core capabilities in Maritime Operations AI:
1. **YOLO26s SAR Vessel Detector**: Spaceborne Sentinel-1 native Single Look Complex (SLC) satellite vessel detection.
2. **Temporal Fusion Transformer (TFT) Vessel Performance Surrogate**: Deep-learning speed degradation surrogate under environmental wind, wave, and swell conditions.

Both training scripts are **strictly CUDA-GPU enforced** to run on a free **NVIDIA Tesla T4 GPU** on Kaggle. Mac MPS or CPU training is blocked by project policy.

---

## 1. Directory Structure

```
ml/kaggle/
├── train_yolo26s.py              # YOLO26s Kaggle GPU training script (CUDA-enforced, AMP, radar augmentations)
├── train_tft.py                  # Vessel Performance TFT Kaggle GPU training script (CUDA-enforced, AdamW)
├── dataset_check.py              # Pre-flight audit, 5-col label format check, GT visualization
├── export_model.py               # Evaluation on held-out test set, prediction overlays, metadata export
├── requirements.txt              # Python dependencies
├── opensar_slc_corrected.zip     # Active SAR Dataset: 3,573 native 512x512 SLC images (699 MB, zero hash leaks)
└── README.md                     # Step-by-step Kaggle execution guide
```

---

## 2. ML Pipeline Audit Findings & Fixes Summary

### 2.1 TFT Vessel-Performance Pipeline Audit
- **Target Leakage Findings**:
  - No future trajectory variables (`fut_x`, `fut_y`, `fut_interp`) leak into historical feature sequences.
  - Normalization statistics are strictly computed on the training split only (`compute_normalization_stats`).
  - Vessel MMSIs are disjoint across splits (Train: 2,727 MMSIs, Val: 317 MMSIs, Test: 335 MMSIs).
  - `met_wind_rel_heading_deg` is relative to historical course over ground (`last_cog`), not future heading.
  - **Identified Autoregressive Shortcut**: Cruising vessels in open transit maintain nearly constant speed (median speed CV is 1.2%). As a result, historical speed (`hist_sog[-1]` and `mean_sog`) has a **0.989** correlation with future displacement speed (`target_sog = (fut_displacement_m / 1852.0) * 6.0`). Feeding `hist_displacement_m` and constant flags (`core_eligible=1.0`, `full_eligible=1.0`) into `vessel_stats` caused the model to learn a trivial identity shortcut rather than hydrodynamic resistance.
- **Implemented Fixes**:
  - Removed uninformative constant eligibility flags (`core_eligible`, `full_eligible`).
  - Replaced the direct displacement shortcut with operational speed dynamics: `[mean_sog, min_sog, max_sog, std_sog]` over the observation window.
  - Created a dedicated, standalone Kaggle GPU training script (`ml/kaggle/train_tft.py`) with CUDA enforcement, early stopping, and automatic held-out test evaluation.

### 2.2 YOLO26s SAR Native SLC Dataset Architecture & Audit
- **Why Native SLC Replaced GRD**:
  - Source OpenSAR XML `<BoundingBox>` annotations are defined exclusively in the $512 \times 512$ SLC radar frame (`Scene_Sample`, `Scene_Line`).
  - GRD patches have non-square, variable dimensions (e.g. $203 \times 814$, $203 \times 1334$) and suffer from swath boundary truncation. Attempting to linearly stretch $512 \times 512$ coordinates onto truncated GRD rasters caused bounding-box errors of up to $288\text{ pixels}$.
  - The active dataset was rebuilt directly from native Sentinel-1 $512 \times 512$ SLC complex float32 rasters (`_SLC_VV.tiff`, `_SLC_VH.tiff`). Magnitude calculation $|I + j Q|$ and deterministic 2nd-to-99.5th percentile contrast stretching produces true 1:1, pixel-exact registration.
- **Confidence Filtering & Hard Negatives**:
  - Unreliable annotations marked `<Confidence>LOW</Confidence>` (underway moving vessels with 20–40 pixel Doppler azimuth shifts relative to uncorrected AIS, such as `DB_OPENSAR_VD_109`) are completely excluded from the dataset (never converted to background negatives) to avoid penalizing the model for detecting unannotated vessels.
  - Clean ocean patches ($Number\_of\_ships = 0$) are fully preserved as hard negatives (2,120 background images) to suppress sea-clutter false alarms.
  - Cross-split duplicate image hashes were audited; 1 duplicate background patch in train matching test was purged.
- **Exclusion & Reconciliation Bookkeeping**:
  - Source archive patches: **4,047** (2,671 train, 311 val, 1,065 test).
  - Low-confidence / misaligned patch exclusions: **473** (315 train, 42 val, 116 test).
  - Additional cross-split duplicate removed from training: **1** (`DB_OPENSAR_VD_5875`).
  - Total excluded patches: **474** ($473 + 1$).
  - Final dataset image count: **3,573** ($4,047 - 474$).
- **Final Active Dataset Composition**:
  - **Train**: 2,355 images (951 vessel, 1,404 background), 2,084 valid vessel annotations.
  - **Validation**: 269 images (108 vessel, 161 background), 197 valid vessel annotations.
  - **Held-Out Test**: 949 images (394 vessel, 555 background), 1,220 valid vessel annotations.
  - **Total**: 3,573 images ($2,355 + 269 + 949$), 3,501 valid vessel annotations, 0 format errors, 0 cross-split hash leaks.

---

## 3. Step-by-Step Kaggle GPU Execution Guide

### Option 1: Training YOLO26s SAR Vessel Detector on Kaggle GPU

#### Step 1: Create a Kaggle Notebook
1. Go to [Kaggle](https://www.kaggle.com/) and click **Create** -> **New Notebook**.
2. Name the notebook: `maritime-yolo26s-sar-training`.

#### Step 2: Enable GPU Accelerator
1. In the right sidebar under **Notebook options**:
   - Set **Accelerator** to **GPU T4 x1**.
   - Ensure **Internet** is toggled **ON** (needed for Ultralytics base weights and dependencies).

#### Step 3: Upload the Code & Dataset
1. Go to **Datasets** -> **New Dataset** -> Upload `ml/kaggle/opensar_slc_corrected.zip` (named `opensar-slc-corrected`).
2. Attach it to your notebook (mounted at `/kaggle/input/opensar-slc-corrected/opensar_slc_corrected` or root).
3. Upload `train_yolo26s.py`, `dataset_check.py`, `export_model.py`, and `requirements.txt` to the notebook root.

#### Step 4: Run Notebook Cells

**Cell 1: Environment Setup**
```bash
!pip install -q -r requirements.txt || pip install -q ultralytics pyyaml pillow matplotlib
```

**Cell 2: Verify GPU**
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Model:     ", torch.cuda.get_device_name(0))
    print("VRAM Available:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")
else:
    raise RuntimeError("Please enable GPU accelerator (GPU T4) in notebook settings!")
```

**Cell 3: Automatic Dataset Resolution & Pre-Flight Audit**
```bash
# dataset_check automatically locates the dataset directory or extracts opensar_slc_corrected.zip if uncompressed
!python dataset_check.py --output-yaml dataset_run.yaml --vis-dir ground_truth_samples
```

**Cell 4: Launch YOLO26s Training on CUDA GPU**
```bash
!python train_yolo26s.py \
    --epochs 150 \
    --batch-size 16 \
    --imgsz 640 \
    --patience 25 \
    --pretrained yolo26s.pt \
    --project-dir kaggle_runs \
    --export-dir export
```

**Cell 5: Download Artifacts**
```python
from IPython.display import FileLink
FileLink("yolo26s_kaggle_artifacts.zip")
```

---

### Option 2: Training Vessel Performance TFT on Kaggle GPU

#### Step 1: Create a Kaggle Notebook
1. Create a new notebook named: `maritime-tft-performance-training`.
2. Set **Accelerator** to **GPU T4 x1** in the right sidebar.

#### Step 2: Upload NOAA Track A Dataset
1. Zip `data/training/noaa_track_v1/` and upload to Kaggle as a dataset named `noaa-track-v1`.
2. Attach it to your notebook (available at `/kaggle/input/noaa-track-v1/noaa_track_v1`).
3. Upload `train_tft.py` to `/kaggle/working/`.

#### Step 3: Run Notebook Cells

**Cell 1: Environment Setup**
```bash
!pip install -q torch numpy
```

**Cell 2: Verify GPU**
```python
import torch
assert torch.cuda.is_available(), "CUDA GPU required! Enable GPU T4 in settings."
print("GPU:", torch.cuda.get_device_name(0))
```

**Cell 3: Launch TFT Training**
```bash
!python train_tft.py \
    --data-dir /kaggle/input/noaa-track-v1 \
    --epochs 35 \
    --batch-size 128 \
    --lr 0.001 \
    --patience 6 \
    --export-dir export_tft
```

**Cell 4: Download Artifacts**
```python
from IPython.display import FileLink
FileLink("tft_kaggle_artifacts.zip")
```

---

## 4. Local Deployment of Trained Artifacts

Once training completes on Kaggle and artifacts are downloaded:

### Integrating YOLO26s:
1. Unzip `yolo26s_kaggle_artifacts.zip`.
2. Copy `best.pt` to:
   - `ml/weights/yolo26s_sar_vessel.pt`
   - `backend/ml/weights/yolo26s_sar_vessel.pt`
3. Copy `yolo26s_metadata.json` to:
   - `ml/weights/yolo26s_metadata.json`
   - `backend/ml/weights/yolo26s_metadata.json`

### Integrating TFT:
1. Unzip `tft_kaggle_artifacts.zip`.
2. Copy `tft_vessel_performance.pt` to:
   - `ml/weights/tft_vessel_performance.pt`
   - `backend/ml/weights/tft_vessel_performance.pt`
3. Copy `tft_metadata.json` to:
   - `ml/weights/tft_metadata.json`
   - `backend/ml/weights/tft_metadata.json`
