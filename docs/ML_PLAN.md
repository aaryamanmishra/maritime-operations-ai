# Machine Learning & Computer Vision Plan (Revised)

## 1. Domain Scope & ML Responsibilities

The Machine Learning subsystem delivers two primary analytical capabilities:
1. **SAR Satellite Vessel Detection**: Modern lightweight SAR ship detector targeting Copernicus Sentinel-1 Level-1 GRD imagery to localize maritime surface vessels.
2. **Deep-Learning Vessel Performance Surrogate**: Temporal deep-learning model (`domain.ml_performance`) predicting vessel speed degradation / power surge under environmental weather stress, feeding the routing optimization engine.

---

## 2. ML Data Provenance & Pre-Training Audit

To avoid blindly downloading multi-gigabyte or unverified datasets, an explicit ML Data Audit is mandatory prior to training.

### 2.1 Candidate SAR Detection Datasets Audit Matrix

| Dataset | Provider / Source | Geo & Temporal Coverage | Vessel Types | Labels & BBox Format | Resolution / Sensor | License & Redistribution | Size | Target Suitability & Leakage Risks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LS-SSDD-v1.0** | Key Laboratory of Digital Earth, CAS | Global coastal & open ocean (2015–2020) | Bulk, container, tanker, fishing | Horizontal & Oriented Bounding Boxes (YOLO/VOC) | 5m–20m (Sentinel-1 IW / TerraSAR-X) | Academic / Non-commercial research | ~1.5 GB | **High suitability**. Realistic ocean clutter and coastal rocks. Preprocessing: chip slicing. Leakage risk: coastal land overlap. |
| **HRSID** | University of Electronic Science & Technology | Diverse global maritime regions (2015–2020) | Diverse commercial vessels | 16,951 bounding boxes (MS-COCO format) | 0.5m–3m (Sentinel-1B, TerraSAR-X) | Academic research open access | ~4.0 GB | **High suitability**. High-resolution chips. Preprocessing: multi-polarization (VV/VH) channel extraction. |
| **xView3-SAR** | Defense Innovation Unit (DIU) / Global Fishing Watch | Global maritime exclusive economic zones (2019–2021) | Fishing, cargo, tanker, passenger | Bounding boxes, vessel length, AIS match label | 10m–20m (Sentinel-1 IW GRD) | Creative Commons CC-BY-NC-SA 4.0 | ~150 GB (Full) / ~10 GB (Dev split) | **Highest fidelity for operational use**, but large size. Dev split must be audited for scene duplication across splits. |

### 2.2 Candidate AIS Vessel Performance Datasets Audit Matrix

| Dataset | Provider / Source | Geo & Temporal Coverage | Features Included | Target Variable | License | Target Suitability & Leakage Risks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Danish Maritime Authority (DMA) Open AIS + ERA5/Open-Meteo** | Danish Maritime Authority / Copernicus | North Sea / Baltic Sea / Kattegat (2021–2023) | MMSI, dimensions, SOG, COG, draft, wave height, wave period, current, wind | Actual speed degradation ($\Delta V = V_{calm} - SOG$) | Open Public Data (Creative Commons / Public Domain) | **High suitability**. Dense verified tracks with high weather variability. Leakage risk: GPS smoothing artifacts near ports. |
| **MarineCadastre Open AIS (NOAA/BOEM)** | NOAA Office for Coastal Management | US Coastal Waters & Atlantic/Pacific corridors | MMSI, vessel type, SOG, COG, heading, draught | Speed loss under coastal/oceanic currents | US Public Domain | **High suitability**. Very clean time series; requires spatial joins with historical marine weather grids. |

---

## 3. SAR Vessel Detector: Architecture & Training Results (Implemented — Phase 4)

### 3.1 Model Architecture & Selection
- **Selected Architecture**: `YOLO26s` (Ultralytics / OpenSAR Insight) with keypoint centroid representation (`kpt_shape: [1, 3]`).
- **Rationale**:
  - Purpose-built for spaceborne SAR imagery where ship targets appear as compact high-intensity backscatter blips against speckle noise and sea clutter.
  - Lightweight single model: 9,751,752 parameters (20.6 MB weights file).
  - High computational efficiency for tiled sliding-window inference across Sentinel-1 swaths.

### 3.2 Training Configuration & Dataset
- **Dataset**: OpenSAR Insight Sentinel-1 GRD filtered dataset (`https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset`).
- **Filtered Subset**: Verified high-contrast annotations from `yolo_dataset_filtered.csv` with standard 5-column detection labels (`0 xc yc w h`):
  - **Train**: 203 patches (349 vessels)
  - **Validation**: 13 patches (21 vessels)
  - **Held-Out Test**: 100 patches (257 vessels)
- **Model Architecture**: `YOLO26s` standard object detection (`task: detect`, `nc: 1`, class 0 = vessel).
- **Training Setup**:
  - Script: `ml/kaggle/train_yolo26s.py`
  - Platform: Kaggle NVIDIA Tesla T4 GPU (CUDA `device=0`, PyTorch FP16 Automatic Mixed Precision `amp=True`)
  - Image size (`imgsz`): 640
  - Batch size: 16
  - Augmentations: SAR radar-specific (no color jitter: `hsv=0`, `fliplr=0.5`, `flipud=0.5`, `mosaic=0.5`)
  - Target epochs: 150 (Early stopping triggered at epoch 87 with patience 25)
  - Epochs completed: 87
  - Best epoch: 62

### 3.3 Evaluation Results & Operational Benchmarks
- **Held-Out Test Split (100 patches, 257 vessels)**:
  - **Model Size**: `20.6 MB` (9.75M parameters)
  - **Measured CPU Inference Latency**: `57.10 ms` per $640 \times 640$ tile
  - **Precision**: `0.4044` (40.44%)
  - **Recall**: `0.3152` (31.52%)
  - **mAP50**: `0.2335` (23.35%)
  - **mAP50-95**: `0.0471` (4.71%)
- **Generalization Gap**:
  - Validation metrics achieved mAP50 of ~0.515, while the strictly held-out test split evaluated at mAP50 of 0.2335.
  - This generalization gap reflects challenging radar noise regimes, speckle variance, variable satellite incidence angles, and ocean surface clutter in complex coastal test scenes.
- **Operational Status**:
  - Declared strictly as `TRAINED / OPERATIONAL` (never "certified" or "production-validated").
  - Detections are probabilistic operational estimates intended to focus maritime operator attention, not certified ground truth.
- **Artifacts**:
  - `ml/weights/yolo26s_sar_vessel.pt` & `backend/ml/weights/yolo26s_sar_vessel.pt`
  - `ml/weights/yolo26s_metadata.json` & `backend/ml/weights/yolo26s_metadata.json`

---

## 4. Deep-Learning Vessel Performance Model (Implemented — Phase 3)

### 4.1 Architecture & Implementation
- Located in `ml/models/tft.py` and `backend/app/domain/ml_performance/tft.py`.
- **Architecture**: `VesselPerformanceTFT` (149,825 parameters):
  - **Static Covariate Encoders**: Vessel dimensions (`length_m`, `beam_m`, `draft_m`, `displacement_t`, `design_speed_kn`).
  - **Temporal Past Covariates**: 6-step history (60-minute window) of past speed, course, and distance displacement.
  - **Temporal Future/Environmental Covariates**: Significant wave height ($H_s$), wave direction, wave period, 10m wind speed, wind direction, ocean current velocity, current direction, sea surface temperature, and swell height.
  - **Interpretable Attention & Gating**: Gated Residual Networks (GRN) with Variable Selection Networks (VSN) and Interpretable Multi-Head Attention.
- **Dataset**: MARIS-Forecast NOAA Track A (`10.5281/zenodo.21224009` / `mark000071/envship_v2_datasets`):
  - 48,000 train, 6,000 val, 6,000 test samples (vessel-disjoint).
  - Pre-split archive size: 53.11 MB compressed.
- **Validation Results on Unseen Test Set (335 vessels)**:
  - **Test MAE**: `0.230 knots`
  - **Test RMSE**: `0.446 knots`
  - **Test $R^2$**: `0.9899`
  - **Inference Latency**: `1.676 ms` on CPU.
- **Production Integration**: Loaded as a singleton in `backend/app/domain/ml_performance/inference.py` (`TFTPerformancePredictor`), producing predicted actual SOG and speed loss ($\Delta V$) combined with Kwon wave degradation for the weather routing optimizer.

---

## 5. AIS Dark-Vessel Spatiotemporal Correlation Engine

When a Sentinel-1 scene is acquired at timestamp $T_{sat}$, each detected target bounding box has center $(lon_{det}, lat_{det})$.

### 5.1 Correlation Protocol
1. **Spatial & Temporal PostGIS Query**: Search `current_vessel_state` and `vessel_position_history` within radius $R = 3000\text{ m}$ and temporal window $[T_{sat} - 15\text{ min}, T_{sat} + 15\text{ min}]$ using PostGIS geography calculation `ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, 3000)`.
2. **Correlation Outcome**:
   - **`AIS-MATCHED`**: Target correlated with an active AIS vessel within distance threshold. Records matched MMSI, distance in meters, and time delta in seconds.
   - **`AIS-UNMATCHED`**: No active AIS candidate within spatiotemporal bounds. Records explicit explanation: `"No AIS broadcast found within ±15m and 3.0km"`.
3. **Operational Ethics Guardrail**:
   - Non-reporting contacts are strictly labeled as `AIS-UNMATCHED`.
   - Never designated as "illegal vessels", "illegal ships", or "criminal vessels".
   - Radar contacts are explicit timestamped snapshot observations from the satellite pass epoch, never presented as "live radar".
