# External Data Sources, Licensing, and Provenance

This document establishes the official provenance, intellectual property licensing, authentication requirements, rate constraints, and fallback plans for all external data sources integrated into **Maritime Operations AI**.

---

## 1. Summary of External Services

| Service / Dataset | Modality | Provider | License / Terms of Service | Auth / Keys | Cost / Limits | Fallback Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AISStream.io** | Live AIS Telemetry | AISStream Team | Free Community Tier for personal/educational projects | Free API Key required | Free; global stream with bounding-box filtering | Norwegian Coastal Admin (Kystverket) Open AIS / Danish Maritime Authority Open AIS data dumps |
| **Open-Meteo Marine API** | Waves, Currents, Wind Forecasts | Open-Meteo GmbH | Non-commercial / Open-Source (Attribution required, CC-BY 4.0) | None required | Free up to 10,000 daily API calls; 5,000/hour | NOAA Global WaveWatch III / GFS public GRIB2 dumps via AWS Open Data |
| **MARIS-Forecast (NOAA Track A)** | AIS Trajectory & Environmental Co-located Dataset | MARIS-Forecast / Zenodo (`10.5281/zenodo.21224009`) / Hugging Face (`mark000071/envship_v2_datasets`) | Composite: NOAA AIS (U.S. Public Domain) + Release (CC-BY 4.0) + OSM (ODbL) | None required | Free open download (~53.1 MB targeted splits: train/val/test) | Pre-trained TFT checkpoint (`tft_vessel_performance.pt`) |
| **Copernicus Data Space Ecosystem (CDSE)** | Sentinel-1 SAR Radar Imagery | European Space Agency (ESA) / EC | Open Access (Creative Commons CC-BY-SA 3.0 IGO) | Free CDSE Account (OAuth2) | Free; 4 concurrent downloads, generous daily bandwidth | Pre-packaged open SAR scene archives (Kaggle / Zenodo) |
| **Searoute / Global Maritime Network** | Maritime Route Graph & Choke Points | Searoute-py / Eurostat / EMODnet | Open Source (MIT / CC-BY 4.0) | None | Local static GeoJSON / NetworkX graph file | World Maritime Routes GeoJSON; local PostGIS Dijkstra network |
| **LS-SSDD-v1.0 & HRSID** | SAR Vessel Training Benchmarks | Institute of Remote Sensing / Open Academic | Academic / Research Open Access | None | Open Download (Baidu / Google Drive / Zenodo) | xView3-SAR Maritime Challenge Dataset (DIU) |
| **Natural Earth / GSHHG** | Global Coastline & Land Polygons | Natural Earth / NOAA GSHHG | Public Domain (CC0) | None | Local vector file / PostGIS spatial table | OpenStreetMap Shoreline Extract |
| **Carto Positron / OpenFreeMap** | Base Vector Map Tiles | CARTO / OpenFreeMap | OpenStreetMap Attribution (ODbL) | None required | Free vector tile hosting | Local PMTiles file hosted directly via Nginx |

---

## 2. In-Depth Provenance Specifications

### 2.1 AISStream (Live AIS)
- **Data Attributes**: MMSI, Vessel Name, Call Sign, IMO, Ship Type, Navigational Status, Latitude, Longitude, SOG (Speed Over Ground), COG (Course Over Ground), True Heading, Rate of Turn, UTC Timestamp.
- **Licensing Restrictions**: Free personal and educational use. Redistribution of bulk raw feed is prohibited. Application must filter by region bounding box to minimize server load.
- **Protocol**: Real-time secure WebSocket (`wss://stream.aisstream.io/v0/stream`). Subscription payload is submitted immediately upon connection containing API key, message type filter (`PositionReport`, `StandardClassBPositionReport`, `ShipStaticData`), and bounding box coordinates.
- **Initial Development Region**: English Channel / Dover Strait (`AIS_BBOX_MIN_LAT=49.5`, `AIS_BBOX_MIN_LON=-2.0`, `AIS_BBOX_MAX_LAT=51.5`, `AIS_BBOX_MAX_LON=2.5`), providing high-density commercial traffic without saturating development bandwidth.
- **Handling Strategy**: The backend worker connects with client-defined bounding boxes covering active operating regions, avoiding unnecessary global saturation. The worker auto-reconnects with exponential backoff and randomized jitter on network disconnects.

### 2.2 Open-Meteo Marine API
- **Data Attributes**: Wave height ($H_s$), wave direction, wave period, surface ocean current velocity, current direction, 10m wind speed, wind direction.
- **Resolution**: 0.08° to 0.25° grid (~5 to 25 km), hourly steps out to 7 days.
- **Licensing Restrictions**: Attribution required: *"Marine weather forecasts provided by Open-Meteo.com"*.
- **Handling Strategy**: Periodic sync worker fetches gridded forecast bounding boxes every 4 hours, caching time-step matrices into Redis. Routing queries read directly from cache, consuming less than 100 API calls per day (well within the 10,000/day free limit).

### 2.3 Copernicus Data Space Ecosystem (Sentinel-1 SAR)
- **Data Attributes**: C-band Synthetic Aperture Radar Level-1 Ground Range Detected (GRD) in Interferometric Wide (IW) swath mode. Polarizations: VV and VH.
- **Licensing Restrictions**: Free and open access for any lawful purpose under EU Copernicus Sentinels Data Policy.
- **Handling Strategy**: User selects a geographic area of interest (AOI) and date range. The worker searches Sentinel-1 footprints via the STAC API, downloads cropped raster chips (or preview quicklooks), and processes radar backscatter intensities without storing multi-terabyte raw SAFE archives permanently.

### 2.4 Maritime Routing Graph
- **Data Attributes**: Graph nodes (ports, waypoints, navigational waypoints) and edges (distance in nautical miles, draft restrictions, canal choke points such as Suez, Panama, Malacca).
- **Licensing Restrictions**: MIT License (Searoute algorithm and Eurostat maritime geometry network).
- **Handling Strategy**: Graph is pre-computed and stored locally in PostGIS tables (`routing_nodes`, `routing_edges`) and indexed in-memory using NetworkX for instantaneous pathfinding.

### 2.5 MARIS-Forecast (NOAA Track A) Benchmark Dataset
- **Upstream Repository**: `https://github.com/mark000071/MARIS-Forecast`
- **Current Zenodo DOI**: `10.5281/zenodo.21224009`
- **Hugging Face Mirror**: `https://huggingface.co/datasets/mark000071/envship_v2_datasets`
- **Target Sub-archive**: Track A — United States NOAA (`data/training/noaa_track_v1/`)
- **Acquired Files**:
  - `train/part-000.csv.gz` (42.6 MB, 48,000 samples, 2,727 distinct vessels)
  - `val/part-000.csv.gz` (5.2 MB, 6,000 samples, 317 distinct vessels)
  - `test/part-000.csv.gz` (5.2 MB, 6,000 samples, 335 distinct vessels)
  - Total targeted acquisition: 53.11 MB (avoiding full 49.1 GB raw release)
  - Samples are strictly vessel-disjoint across splits.
- **Composite Licensing**:
  - NOAA AIS Trajectory Component: U.S. Public Domain.
  - MARIS-Forecast Release: Creative Commons Attribution 4.0 International (CC-BY 4.0).
  - OpenStreetMap Coastline/Bathymetry Elements: Open Database License (ODbL).
  - Atmospheric & Oceanographic Co-located Fields: ECMWF / GFS upstream attribution.
- **Model Output Disclaimer**:
  - Predictions from the Temporal Fusion Transformer (TFT) estimate vessel speed degradation under wind, waves, and surface currents.
  - Energy and fuel values are calculated via transparent naval architecture equations (Holtrop-Mennen calm water resistance, Blendermann wind resistance, Kwon wave resistance, and IMO Fourth GHG Study SFOC curves).
  - All fuel and energy outputs are strictly labeled as **ESTIMATES**. No claims of certified savings, guaranteed navigational safety, or certified nautical optimality are made.

### 2.6 OpenSAR Insight Sentinel-1 Vessel Detection Dataset
- **Upstream Repository**: `https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset`
- **Modality**: C-band Sentinel-1 Level-1 GRD Interferometric Wide (IW) dual-polarization (VV/VH), tiled into 640x640 inference patches with standard 5-column YOLO 2D detection format (`0 xc yc w h`).
- **Target Sub-archive Acquired**:
  - `yolo_dataset_filtered.csv` (authoritative filtered subset of verified, high-quality vessel patches)
  - `train_patches.tar.zst`, `val_patches.tar.zst`, `test_patches.tar.zst`
  - `train_labels.tar.zst`, `val_labels.tar.zst`, `test_labels.tar.zst`
  - Total filtered dataset: 203 train patches (349 vessels), 13 val patches (21 vessels), 100 test patches (257 vessels).
- **Composite Licensing**:
  - OpenSAR Insight Code & Metadata: MIT License.
  - Underlying Radar Imagery: European Union Copernicus Open Access (Creative Commons CC-BY-SA 3.0 IGO).
  - YOLO Framework: Ultralytics AGPL-3.0.
- **Model Checkpoint & Training Provenance**:
  - Model: `YOLO26s` standard object detector (`task: detect`, `nc: 1`, class 0 = vessel).
  - Weights: `ml/weights/yolo26s_sar_vessel.pt` & `backend/ml/weights/yolo26s_sar_vessel.pt`.
  - Training Platform: Kaggle NVIDIA Tesla T4 GPU (CUDA `device=0`, FP16 AMP).
  - Completed Epochs: 87 (Best epoch: 62, early stopping patience 25).
  - Held-out Test Metrics (100 images, 257 vessels): Precision: 0.4044, Recall: 0.3152, mAP50: 0.2335, mAP50-95: 0.0471.
  - Measured local CPU inference latency: 57.10 ms per 640x640 tile.
  - Operational Status: `TRAINED / OPERATIONAL` (Explicitly not certified or production-validated).
  - Generalization Gap: Validation mAP50 was ~0.515 across the 13 validation patches, while the 100-image held-out test split evaluated at 0.2335 mAP50. This gap reflects real-world variability in radar speckle noise, high-clutter sea states, and variable incidence angles in diverse coastal waters.
- **Operational Ethics & Non-Prejudicial Classification**:
  - Detected radar targets are strictly classified as either `AIS-MATCHED` (correlated with an AIS broadcast within $\pm 15$ min and $3.0\text{ km}$) or `AIS-UNMATCHED` (no correlation).
  - Prejudicial terminology such as "illegal vessel" or "criminal ship" is strictly prohibited. Unmatched radar contacts represent non-broadcasting targets under configured correlation rules and may include non-SOLAS vessels, small crafts, or transient coverage gaps.

---

## 3. Strict Compliance Guidelines

1. **No Data Fabrication**: Simulated or synthetic vessel coordinates must never be passed off as authentic live AIS. If external network access is unavailable, the user interface clearly renders an offline / disconnected banner.
2. **Attribution Display**: The application footer and "About" dialog must display explicit attribution to OpenStreetMap, Open-Meteo, Copernicus ESA, and AISStream.
3. **Privacy & Security**: MMSI numbers and public radio broadcasts are non-confidential international open radio transmissions; however, no personal crew or vessel owner privacy-invasive lookups are executed.
