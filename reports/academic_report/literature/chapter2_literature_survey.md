# Chapter 2: Literature Survey

## 2.1 Existing Systems and Their Limitations

Maritime surveillance has evolved over several decades from visual watchkeeping and coastal radar towers to automated satellite remote sensing and radio broadcast tracking. However, existing commercial and academic systems suffer from structural limitations that prevent reliable, all-weather identification of non-cooperative vessels.

### 2.1.1 Classical Radar Ship Detection: The CFAR Family
Prior to the advent of deep learning in remote sensing, radar target detection relied almost exclusively on statistical Constant False Alarm Rate (CFAR) algorithms [1]. The foundational premise of CFAR is to dynamically estimate the statistical probability density function (PDF) of ocean clutter in a local sliding window surrounding a target cell under test (CUT), setting an adaptive threshold to maintain a constant false alarm probability:

$$T = \alpha \cdot \hat{\mu}_{clutter}$$

where $\hat{\mu}_{clutter}$ is the estimated clutter power and $\alpha$ is a threshold multiplier derived from assumed sea clutter distributions (e.g., Gaussian, Rayleigh, Weibull, K-distribution). Multiple CFAR variants were engineered to handle complex maritime environments:
- **Cell-Averaging CFAR (CA-CFAR):** Averages all reference cells in leading and lagging windows. While optimal in homogeneous open-ocean clutter, CA-CFAR suffers catastrophic masking ("target masking") in dense maritime traffic where neighboring ships contaminate the reference cells, artificially inflating the threshold and causing missed detections.
- **Greatest-Of CFAR (GO-CFAR) & Smallest-Of CFAR (SO-CFAR):** Designed for clutter edges (e.g., ocean-land boundaries). GO-CFAR reduces false alarms at boundaries but exacerbates target masking; SO-CFAR improves multi-target resolution but suffers high false alarm rates along sea-clutter boundaries.
- **Order-Statistic CFAR (OS-CFAR):** Sorts reference cells in ascending order and selects the $k$-th percentile value. While resilient against interfering targets, OS-CFAR requires heavy sorting computation and struggles with irregular coastal geometry.

#### Critical Limitations of CFAR in Operational Maritime AI:
1. **High Inshore False Alarm Rate:** Breaking waves, coastal headlands, sandbars, and tidal rips exhibit severe radar backscatter spikes that mimic ship returns, generating hundreds of false positives in coastal zones.
2. **Empirical Parameter Fragility:** CFAR algorithms require manual tuning of guard cells, reference windows, and threshold scaling factors for each specific sea state, incidence angle, and wind condition, making global automated operations infeasible.
3. **Absence of Semantic Feature Extraction:** CFAR operates purely on single-pixel intensity statistics, failing to exploit structural, morphological, aspect-ratio, and contextual clues that distinguish ships from radar artifacts.

---

### 2.1.2 Automatic Identification System (AIS) Vulnerabilities
The Automatic Identification System (AIS) was developed by the IMO to prevent vessel collisions at sea by continuously broadcasting VHF radio packets (ITU-R M.1371) containing vessel identity, position, speed, and heading [2]. Despite its ubiquity, AIS suffers from systemic operational vulnerabilities when used for maritime security:
1. **Cooperative Failure & Dark Vessels:** Because transponders require continuous electrical power and deliberate operation, bad actors systematically switch off AIS circuit breakers during illicit activities (e.g., unauthorized fishing in EEZs, ship-to-ship fuel transfers, contraband smuggling).
2. **Identity & Geolocation Spoofing:** AIS messages are unencrypted and lack cryptographic authentication. Vessels can transmit forged Maritime Mobile Service Identities (MMSI), falsified ship names, or synthetic GPS coordinates to create "ghost fleets" thousands of miles from their actual physical locations.
3. **Propagation & Satellite Blackouts:** Terrestrial AIS VHF signals have a line-of-sight range typically restricted to $20\text{--}40\text{ nautical miles}$ from coastal base stations. While Low Earth Orbit (LEO) satellite constellations (e.g., Spire, exactEarth, Orbcomm) capture AIS globally, satellite message collision in dense choke points (e.g., Strait of Malacca, English Channel) causes significant message packet loss and latency deltas reaching 15–45 minutes.

---

### 2.1.3 Satellite Radar Modalities: The SLC vs GRD Domain Gap
Sentinel-1 synthetic aperture radar data is distributed in two primary Level-1 formats that reflect fundamentally distinct signal processing stages:

| Physical / Engineering Characteristic | Single Look Complex (SLC) | Ground Range Detected (GRD) |
| :--- | :--- | :--- |
| **Data Representation** | Complex numbers (In-phase $I$, Quadrature $Q$) | Real positive amplitude / intensity ($DN$) |
| **Geometric Plane** | Slant-Range Geometry (satellite line-of-sight) | Ground-Range Geometry (projected onto WGS84 ellipsoid) |
| **Speckle Statistics** | Single-look, Rayleigh-distributed speckle noise | Multi-looked ($5 \times 1$ in IW mode), Gamma-distributed |
| **Spatial Pixel Spacing** | $2.3\,\text{m}$ (azimuth) $\times 14.1\,\text{m}$ (slant range) | $10.0\,\text{m} \times 10.0\,\text{m}$ square ground range |
| **Pixel Aspect Ratio** | Highly asymmetrical ($\sim 1:6.1$) | Uniform square ($1:1$) |
| **Point Scatterer Signature** | Sharp point-like phase spikes with sidelobes | Averaged, smoothed intensity centroids |

#### The Modality Collapse Hazard:
A major failure mode in previous research is training deep-learning models on open SLC datasets (e.g., OpenSAR Insight [3]) and attempting to deploy them on real-time operational Copernicus GRD swaths. Because SLC imagery possesses high-frequency Rayleigh speckle and an asymmetrical $1:6$ pixel aspect ratio, an SLC-trained detector looks for sharp, single-pixel interference fringes. When applied to multi-looked GRD imagery, the smoothed Gamma-distributed returns cause complete **confidence collapse**, resulting in zero detections and failure of operational surveillance.

---

### 2.1.4 Flaws in Naive Spatial Correlation Buffering
When correlating satellite detections with AIS broadcasts, early academic systems (e.g., Margarit et al. [4]) implemented static circular buffers (e.g., drawing a $3\text{--}5\,\text{km}$ circle around the radar detection and querying AIS pings). 

This static assumption is physically invalid for moving surface vessels:
- A modern container vessel or patrol craft operating at a transit speed of $22\text{ knots}$ ($11.32\,\text{m/s}$) travels **$10.19\text{ km}$** in just 15 minutes.
- If an AIS ping is logged 15 minutes prior to a Sentinel-1 overpass, the ship will be located more than $10\text{ km}$ away from its previous report.
- A static $3\text{ km}$ buffer fails to capture the compliant vessel, falsely classifying it as an "unmatched dark ship," while potentially matching an unrelated slow-moving barge or fishing vessel within the $3\text{ km}$ circle.

---

## 2.2 Proposed System Approach and Innovations

To overcome these structural limitations, Maritime Operations AI establishes a multi-sensor, multi-stage architecture integrating physical sensor principles, modern deep learning, and kinematic probability theory:

```
+-------------------------------------------------------------------------------------------------+
|                                 MARITIME OPERATIONS AI PIPELINE                                 |
+------------------------------------+------------------------------------------------------------+
| 1. Copernicus Sentinel-1 GRD Swath | Multi-look C-band radar, dual-polarization (VV/VH)         |
| 2. Percentile Clip Normalization   | 2.0% - 99.5% dynamic clip, eliminates extreme speckle spikes|
| 3. Native-Resolution Tiler         | 640x640 sliding window, 64px overlap, preserves 10m pixels  |
| 4. Sensor-Matched YOLO Detector    | Anchor-free YOLO11n / YOLOv8s trained on LS-SSDD-v1.0      |
| 5. Conservative NMS Deduplication  | Overlap band IoU >= 0.30 deduplication with twin provenance |
| 6. Piecewise Bilinear GCP Georef   | Sub-pixel interpolation from product GCP grid to WGS84     |
| 7. Kinematic Dead Reckoning        | Extrapolates AIS position via SOG, COG over time delta dt  |
| 8. Positional Uncertainty Budget   | sigma_total = sqrt(sigma_sar^2 + sigma_ais^2 + sigma_motion^2)|
| 9. Gaussian Likelihood Scoring     | z = distance / sigma_total, score = exp(-0.5 * z^2)        |
| 10. Ethical 4-Tier Classification  | AIS-MATCHED, AMBIGUOUS, AIS-UNMATCHED, INSUFFICIENT-COV    |
+------------------------------------+------------------------------------------------------------+
```

### Core Methodological Innovations:
1. **Sensor-Matched Native-Resolution Processing:** Rather than downsampling massive radar swaths ($25,000 \times 16,500$ px), the pipeline tiles the scene at its native $10\,\text{m} \times 10\,\text{m}$ ground range resolution into overlapping $640 \times 640$ chips. This preserves the radar signatures of small ships ($10\text{--}30\,\text{m}$) while restricting memory usage.
2. **Anchor-Free Decoupled YOLO Architecture:** We deploy Ultralytics YOLO11n and YOLOv8s detectors featuring decoupled heads (separating classification probability from bounding box regression) and Task-Aligned Assigner (TAL) [5]. Radar-specific augmentations ($hsv=0$, $fliplr=0.5$, $flipud=0.5$) respect radar symmetry while eliminating synthetic chromatic distortions.
3. **Rigorous GCP Grid Interpolation:** Eliminating crude global 4-corner affine approximations, the georeferencer parses the official Sentinel-1 product Geolocation Grid Points (GCPs). It executes bisection search along image lines and range pixels, performing piecewise bilinear interpolation to account for non-linear Earth curvature and orbital inclination.
4. **Kinematic & Uncertainty-Aware Spatiotemporal Correlation:** Instead of static buffers, our correlation engine performs kinematic dead reckoning using the vessel's reported Speed Over Ground (SOG) and Course Over Ground (COG) over the elapsed time interval $\Delta t = |T_{sat} - T_{ais}|$. It synthesizes a dynamic positional uncertainty budget:
   $$\sigma_{total} = \sqrt{\sigma_{sar}^2 + \sigma_{ais}^2 + \sigma_{motion}^2}$$
   where $\sigma_{sar} \approx 30\,\text{m}$ (satellite georeferencing error), $\sigma_{ais} \approx 15\,\text{m}$ (marine GNSS variance), and $\sigma_{motion} = 0.08 \cdot v \cdot \Delta t + 0.15 \cdot \Delta t$ accounts for vessel maneuverability, turning, and acceleration uncertainty.
5. **Ethical Four-Tier Classification & Regional Coverage Validation:**
   - `AIS-MATCHED`: Target correlated within hard gate and $z \le 2.5$.
   - `AMBIGUOUS / REVIEW-REQUIRED`: Soft-edge boundary candidate ($2.5 < z \le 4.0$ or distance near gate), flagged for tactical analyst review.
   - `AIS-UNMATCHED`: Potential dark-vessel candidate, validated by confirming active AIS receiver coverage within a $35\text{ km}$ regional radius.
   - `INSUFFICIENT-COVERAGE`: No AIS pings detected across the entire $35\text{ km}$ sector, correctly tagging the detection as an operational receiver blindspot rather than an illegal actor.

---

## 2.3 Benchmark Datasets for SAR Vessel Detection

### 2.3.1 LS-SSDD-v1.0 (Large-Scale SAR Ship Detection Dataset)
Constructed by the Key Laboratory of Digital Earth, Chinese Academy of Sciences (Zhang et al., MDPI Remote Sensing 2020) [6], LS-SSDD-v1.0 is the premier open benchmark for large-scale Sentinel-1 ship detection.
- **Sensor & Coverage:** 15 large-scale Sentinel-1 SAR swaths acquired in Interferometric Wide (IW) swath mode with dual-polarization (VV/VH).
- **Chip Inventory:** 9,000 native-resolution $800 \times 800$ pixel radar chips covering $1,076\,\text{km} \times 1,280\,\text{km}$ total ocean surface.
- **Target Annotations:** 9,000 Pascal VOC XML annotation files containing 15,487 ship targets.
- **Official Split Strategy:**
  - **Training Split:** 5,100 images (derived strictly from Scenes 01 to 10).
  - **Validation Split:** 900 images (derived strictly from Scenes 01 to 10).
  - **Held-Out Test Split:** 3,000 images (derived strictly from Scenes 11 to 15).
  - **Scene-Level Holdout:** Scenes 11 to 15 are strictly isolated from training and validation, ensuring zero cross-scene chip leakage.
- **Dedicated Clutter Subsets:** The 3,000-image test set provides dedicated subsets for **inshore** (coastal, ports, dense islands, high radar clutter) and **offshore** (open ocean, low clutter).

### 2.3.2 OpenSAR Insight Benchmark
Developed by Shanghai Jiao Tong University and hosted on HuggingFace [3], OpenSAR Insight provides Sentinel-1 Single Look Complex (SLC) chips targeting commercial ship types. While valuable for slant-range radar research, its complex-valued speckle signatures exhibit an operational domain mismatch against standard Copernicus Level-1 GRD products.

---

## 2.4 Comparative Literature Analysis

The table below contrasts published ship detection methodologies, datasets, architectures, and quantitative results against the Maritime Operations AI implementations:

| Authors & Citation | Methodology / Architecture | Dataset & Modality | Test Split Size | Precision (%) | Recall (%) | $\text{mAP}_{50}$ (%) | Key Findings & Limitations |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Zhang et al. (2020)** [6] | Faster R-CNN (Two-stage, ResNet-50) | LS-SSDD-v1.0 (Sentinel-1 GRD) | 3,000 chips | 77.80% | 67.20% | **71.60%** | SOTA benchmark in paper; two-stage architecture produces high latency (~240ms). |
| **Zhang et al. (2020)** [6] | YOLOv3 (One-stage, DarkNet-53) | LS-SSDD-v1.0 (Sentinel-1 GRD) | 3,000 chips | 75.20% | 64.10% | **69.40%** | Fast one-stage baseline; anchor box rigidity causes false alarms on coastal rocks. |
| **Zhang et al. (2020)** [6] | RetinaNet (ResNet-50 + FPN) | LS-SSDD-v1.0 (Sentinel-1 GRD) | 3,000 chips | 71.40% | 61.20% | **65.80%** | Focal loss mitigates foreground-background imbalance but suffers in dense ports. |
| **Zhang et al. (2020)** [6] | SSD (VGG-16 backbone) | LS-SSDD-v1.0 (Sentinel-1 GRD) | 3,000 chips | 68.10% | 59.40% | **64.30%** | Moderate accuracy; low recall on small vessels (<15m length). |
| **Li et al. (2021)** [3] | OpenSAR Baseline (CNN) | OpenSAR Insight (Sentinel-1 SLC) | 950 chips | 52.55% | 41.64% | **37.92%** | High slant-range speckle; completely collapses when evaluated on GRD swaths. |
| **Margarit et al. (2021)** [4] | CFAR + Static Spatial Buffer | Sentinel-1 + Terrestrial AIS | Operational Swaths | 61.50% | 54.20% | N/A | Static 5km buffer fails for high-speed ships; CFAR fails in coastal zones. |
| **This Work (Prod GRD)** | **YOLO11n (Phase 6 Production)** | **LS-SSDD-v1.0 (Sentinel-1 GRD)** | **3,000 chips** | **76.58%** | **62.42%** | **63.52%** | **Ultra-lightweight (2.6M params); 121ms CPU latency; 133 confirmed detections on real GRD.** |
| **This Work (Candidate)** | **YOLOv8s-LS-SSDD-v1.0** | **LS-SSDD-v1.0 (Sentinel-1 GRD)** | **3,000 chips** | **80.70%** | **61.60%** | **69.00%** | **Matches paper YOLOv3 mAP with +5.5% higher precision; anchor-free TAL head.** |
| **This Work (Deployed SLC)**| **YOLO11s OpenSAR Baseline** | **OpenSAR Insight (Sentinel-1 SLC)**| **950 chips** | **51.74%** | **45.15%** | **39.41%** | **Best SLC model; retained exclusively for opt-in SLC pipeline inference.** |

---

## 2.5 Summary of Findings

1. **Resolution Supremacy:** Native-resolution radar chip extraction ($640 \times 640$ px at 10m ground range) is mandatory for small vessel detection. Downsampling eliminates small targets before deep feature extraction begins.
2. **Architecture Evolution:** Modern decoupled anchor-free architectures (YOLOv8, YOLO11) outperform older anchor-based detectors (YOLOv3, SSD) by eliminating rigid anchor priors, achieving higher precision ($80.7\%$ vs $75.2\%$) which is crucial for suppressing false alarms in high-clutter coastal regimes.
3. **Modality Specialization:** The physical gap between SLC and GRD imagery cannot be ignored. A production surveillance platform must preserve a dedicated GRD detector for standard operations while isolating SLC models for specialized phase-coherent analysis.
4. **Kinematic Correlation Requirement:** Traditional fixed-radius buffers fail moving targets. Incorporating dead reckoning, variable speed extrapolation, and a dynamic 3-component uncertainty budget provides the mathematical foundation necessary to prevent false dark-vessel alerts.
