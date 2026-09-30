# YOLO26s SAR Vessel Detector — Kaggle GPU Training Package

This directory contains the self-contained training package to train the **YOLO26s** SAR vessel detector on a free **NVIDIA Tesla T4 GPU** in Kaggle.

---

## 1. Package Structure

```
ml/kaggle/
├── train_yolo26s.py          # Main autonomous training script (CUDA-enforced, mixed precision AMP)
├── dataset_check.py          # Pre-flight audit, verification of 5-col labels, and GT visualization
├── export_model.py           # Evaluation on held-out test set, prediction overlays, metadata export
├── requirements.txt          # Python dependencies
├── opensar_filtered.zip      # Complete pre-packaged OpenSAR dataset (36 MB)
└── README.md                 # Step-by-step execution guide
```

---

## 2. Dataset Specification

- **Source**: OpenSAR Insight Sentinel-1 GRD SAR Imagery
- **Task**: Standard 2D Object Detection (`task: detect`)
- **Classes**: Exactly 1 class: `0 = vessel`
- **Label Format**: 5-column normalized YOLO format (`0 <x_center> <y_center> <width> <height>`)
- **Preprocessing**: Contrast stretching (2nd to 99.5th percentile) applied to raw digital numbers, preserving high dynamic range radar backscatter without saturation or blanking.
- **Dataset Split**:
  - `train`: 203 images, 349 vessel annotations
  - `val`: 13 images, 21 vessel annotations
  - `test`: 100 images, 257 vessel annotations (strictly held out)

---

## 3. Step-by-Step Kaggle Setup Guide

### Step 1: Create a Kaggle Notebook
1. Open [Kaggle](https://www.kaggle.com/) and click **Create** -> **New Notebook**.
2. Name the notebook: `maritime-yolo26s-sar-training`.

### Step 2: Enable GPU Accelerator
1. In the right-hand panel under **Notebook options** / **Settings**:
   - Set **Accelerator** to **GPU T4 x1** (or **GPU T4 x2**).
   - Verify that **Internet** is toggled **ON** (required to download base weights `yolo26s.pt` and dependencies).

### Step 3: Provide the Dataset & Scripts

#### Option A: Upload as a Kaggle Dataset (Recommended)
1. Go to **Datasets** -> **New Dataset**.
2. Upload `ml/kaggle/opensar_filtered.zip`.
3. Set the title as `opensar-detection`.
4. In your notebook, click **+ Add Input** and attach `opensar-detection`.
   - The dataset will be mounted at `/kaggle/input/opensar-detection/opensar_filtered`.

#### Option B: Direct Upload via Notebook File Browser or Curl
Upload `opensar_filtered.zip`, `train_yolo26s.py`, `dataset_check.py`, and `export_model.py` directly to the `/kaggle/working/` workspace.

---

## 4. Notebook Execution Cells

Run the following cells sequentially in your Kaggle notebook:

### Cell 1: Environment Setup
```python
# Install required dependencies
!pip install -q -r requirements.txt || pip install -q ultralytics pyyaml pillow matplotlib
```

### Cell 2: Verify GPU Environment
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Model:     ", torch.cuda.get_device_name(0))
    print("VRAM Available:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")
else:
    raise RuntimeError("Please enable GPU accelerator (GPU T4) in notebook settings!")
```

### Cell 3: Dataset Unpack (if using zip)
```python
import os, zipfile

if not os.path.exists("opensar_filtered"):
    if os.path.exists("/kaggle/input/opensar-detection/opensar_filtered"):
        print("Using attached Kaggle dataset.")
    elif os.path.exists("opensar_filtered.zip"):
        print("Unzipping local opensar_filtered.zip...")
        with zipfile.ZipFile("opensar_filtered.zip", "r") as zf:
            zf.extractall(".")
```

### Cell 4: Pre-Flight Dataset Audit
```python
# Run data audit and generate ground-truth diagnostic visualizations
!python dataset_check.py --data-dir opensar_filtered --output-yaml dataset_run.yaml --vis-dir ground_truth_samples
```

### Cell 5: Run YOLO26s Training
```python
# Launch training on NVIDIA Tesla T4 with mixed precision FP16 and radar augmentations
!python train_yolo26s.py \
    --data-dir opensar_filtered \
    --epochs 150 \
    --batch-size 16 \
    --imgsz 640 \
    --patience 25 \
    --pretrained yolo26s.pt \
    --project-dir kaggle_runs \
    --export-dir export
```

### Cell 6: Inspect Test Metrics & Download Artifacts
```python
import json
with open("export/yolo26s_metadata.json", "r") as f:
    meta = json.load(f)

print("=== HELD-OUT TEST METRICS ===")
for k, v in meta["test_metrics"].items():
    print(f"  {k}: {v}")

# File link for downloading artifacts
from IPython.display import FileLink
FileLink("yolo26s_kaggle_artifacts.zip")
```

---

## 5. Generated Artifacts

Upon completion, `yolo26s_kaggle_artifacts.zip` will contain:
1. `best.pt`: Best checkpoint based on validation loss and mAP50.
2. `last.pt`: Final checkpoint.
3. `predictions/`: Prediction overlay visualizations on held-out test images.
4. `yolo26s_metadata.json`: Full training provenance, parameters, hardware info, and held-out test metrics.

---

## 6. Local Integration

Once downloaded to your development machine:
1. Copy `best.pt` to:
   - `ml/weights/yolo26s_sar_vessel.pt`
   - `backend/ml/weights/yolo26s_sar_vessel.pt`
2. Copy `yolo26s_metadata.json` to:
   - `backend/ml/weights/yolo26s_metadata.json`
3. Restart the backend container:
   ```bash
   docker compose restart backend
   ```
4. Verify the model endpoint:
   ```bash
   curl -s http://localhost:8000/api/v1/sar/model/status | jq .
   ```
