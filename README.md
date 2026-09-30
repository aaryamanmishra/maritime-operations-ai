# Maritime Operations AI

A production-grade, open-source maritime operations platform engineered for real-time situational awareness, weather-routing optimization, and satellite radar dark-vessel intelligence.

---

## 🎯 Project Purpose & Primary Capabilities

Maritime Operations AI consolidates three foundational maritime operational capabilities into a single interactive map interface:

1. **Live Vessel Traffic**
   - High-throughput AIS position ingestion and kinematic tracking.
   - Clean architectural separation of live state (`current_vessel_state`) from historical trajectories (`vessel_position_history`).
   - Geo-fenced spatial viewport streaming via WebSockets with sub-50ms target benchmark response times.

2. **AI Weather-Aware Routing**
   - Multi-objective voyage optimization built upon the Eurostat / SeaRoute open maritime navigation network.
   - Deep-learning vessel performance surrogate (candidate: Temporal Fusion Transformer / temporal sequence model) predicting operational speed degradation ($\Delta V$) under marine weather stress.
   - Hydrodynamic vessel resistance and brake power models feeding verified Specific Fuel Oil Consumption (SFOC) engine load curves.
   - *Operational Notice*: All voyage durations, power values, and fuel metrics are explicit computational estimates based on documented engineering models. The system never claims certified fuel savings or guaranteed-safe routing.

3. **Dark-Vessel Intelligence**
   - Event-driven Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) Level-1 GRD scene discovery via STAC catalog queries.
   - Production vessel detector powered by a single trained YOLO26s model on the OpenSAR Insight Sentinel-1 GRD dataset.
   - Spatiotemporal correlation engine comparing radar target detections with recorded AIS baselines within a $\pm 15$-minute window and $3.0\text{ km}$ radius in PostGIS.
   - *Operational Notice*: Satellite radar passes are discrete, timestamped snapshot observations (never "live SAR"). Non-reporting radar contacts are classified strictly as `AIS-UNMATCHED` and correlated contacts as `AIS-MATCHED`. The system never uses prejudicial labels such as "illegal vessel" or "criminal vessel". An AIS-unmatched radar contact is only an observation that did not correlate with available AIS data under the configured matching rules.

---

## 🏛 Architecture Overview

The platform is designed as a **Modular Monolith** with independent background worker processes:
- **Frontend**: React 19, TypeScript, Vite, Nginx reverse proxy.
- **Backend API**: Python 3.12, FastAPI, Pydantic v2 Settings, SQLAlchemy 2.0 Asyncio.
- **Durable Spatial State**: PostgreSQL 16 with PostGIS 3.4 (`geometry(Point, 4326)` & `geometry(LineString, 4326)`).
- **Messaging & Primary Short-Lived Cache**: Redis 7 Alpine (Redis Streams `stream:ais:raw`, Redis Pub/Sub, on-demand Open-Meteo weather caching with 6-hour TTL).
- **Orchestration**: Docker Compose for local zero out-of-pocket development.

---

## 💻 Local Prerequisites

To run Maritime Operations AI locally at zero cost:
- **Operating System**: macOS (Apple Silicon or Intel), Linux, or Windows (WSL2).
- **Docker**: Docker Engine 24+ and Docker Compose v2.20+.
- **RAM**: Minimum 2.5 GB free RAM.
- **Disk Space**: Minimum 5 GB free disk space.
- *Zero paid cloud accounts or subscriptions required.*

---

## 🚀 Environment Setup & Startup Commands

### 1. Clone & Configure Environment
```bash
# Clone the repository
git clone <repo-url> maritime-operations-ai
cd maritime-operations-ai

# Create local environment file from template
cp .env.example .env
```

### 2. Canonical Local Startup (Single Command)
```bash
docker compose up --build
```

### 3. Verify Services
Once started, the services are accessible at:
- **Web UI Application**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Liveness Probe**: [http://localhost:8000/api/health/live](http://localhost:8000/api/health/live)
- **API Readiness Probe**: [http://localhost:8000/api/health/ready](http://localhost:8000/api/health/ready)
- **PostgreSQL / PostGIS**: `localhost:5432` (`user: maritime_user`, `db: maritime_ops`)
- **Redis Broker**: `localhost:6379`

---

## 📁 Repository Structure

```
maritime-operations-ai/
├── README.md                   # Main documentation & quickstart
├── .env.example                # Canonical environment template
├── .gitignore                  # Git exclusions
├── docker-compose.yml          # Local multi-service orchestration
│
├── backend/                    # Python 3.12 FastAPI Modular Monolith
│   ├── Dockerfile              # Python 3.12-slim production container
│   ├── pyproject.toml          # Python dependencies & build config
│   ├── alembic.ini             # Database migration configuration
│   ├── alembic/                # PostGIS Alembic migration scripts
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint, lifespan & middleware
│   │   ├── core/               # Settings, logging, correlation IDs
│   │   ├── api/v1/             # HTTP endpoints (/api/health/live, /ready)
│   │   ├── domain/             # Business logic (vessel_traffic, routing, sar)
│   │   ├── infrastructure/     # Database session & Redis clients
│   │   ├── routing/            # Navigation graph & hydrodynamics
│   │   ├── ml/                 # Model inference interfaces
│   │   └── workers/            # Ingestion, persistence & task workers
│   └── tests/                  # Backend unit, integration & health tests
│
├── frontend/                   # React 19 + TypeScript + Vite
│   ├── Dockerfile              # Multi-stage build (Node 22 builder + Nginx)
│   ├── nginx.conf              # Reverse proxy configuration
│   ├── package.json            # Node dependencies
│   ├── vite.config.ts          # Vite configuration & proxy settings
│   └── src/
│       ├── App.tsx             # Application shell
│       ├── components/         # Header, Navigation, HealthStatus, DashboardShell
│       ├── services/           # Typed API client
│       └── tests/              # Frontend foundation unit tests
│
├── ml/                         # Offline ML training pipelines & data audit
│   ├── data_audit/             # Pre-training dataset audit reports
│   ├── datasets/               # Dataset downloaders (LS-SSDD, HRSID, xView3)
│   ├── training/               # Kaggle/Colab reproducible training scripts
│   └── weights/                # Model artifacts
│
├── data/                       # Maritime graphs, land masks & test scenes
├── scripts/                    # Development scripts & AIS replay tooling
├── tests/                      # Project-wide integration test suites
└── docs/                       # Architecture documents, ADRs & audits
```

---

## 🚦 Implementation Status

| Capability / Layer | Status | Description |
| :--- | :--- | :--- |
| **Phase 1: Foundation** | **COMPLETE** | Docker Compose orchestration, PostGIS migration, Redis client, FastAPI health endpoints, React app shell, automated tests passing. |
| **Phase 2: Vessel Traffic** | **COMPLETE** | Real AISStream WebSocket worker, Redis Stream `stream:ais:raw`, PostGIS `vessel_identity`, `current_vessel_state`, and thinned `vessel_position_history`, WebSocket `/ws/vessels` gateway with viewport filtering, MapLibre GL JS live tracking UI. |
| **Phase 3: Weather Routing** | **COMPLETE** | SeaRoute maritime network, Open-Meteo marine adapter with Redis TTL caching, TFT vessel performance surrogate trained on MARIS-Forecast NOAA Track A (Test MAE: 0.230 kn, R²: 0.9899), transparent physics/SFOC fuel & cost estimation, multi-objective weather-aware route optimizer, interactive unified MapLibre UI. |
| **Phase 4: Dark-Vessel Intel** | Queued | Copernicus STAC discovery, lightweight SAR ship detector, spatiotemporal AIS correlation. |

---

## 🌊 Phase 3: AI Weather-Aware Routing Subsystem

The AI weather-aware routing subsystem computes navigable, fuel-efficient, and weather-conscious maritime transit trajectories between global origins and destinations.

### Architectural Flow
```
Origin / Destination / Departure / Vessel Parameters
                     │
                     ▼
       SeaRoute Base Maritime Network
  (Land-avoiding certified international lanes)
                     │
                     ▼
         Open-Meteo Marine Adapter
    (Hourly waves, currents, wind, SST;
         Redis cache 6h TTL)
                     │
                     ▼
Temporal Fusion Transformer (TFT) Surrogate
  (Trained on MARIS-Forecast NOAA Track A:
   149,825 params, Test MAE: 0.230 knots)
                     │
                     ▼
  Naval Architecture Hydrodynamic Engine
 (Holtrop-Mennen resistance, Blendermann wind,
     Kwon speed loss, IMO Fourth GHG SFOC)
                     │
                     ▼
   Multi-Objective Candidate Optimizer
  (Evaluates base route vs detour corridors)
                     │
                     ▼
Unified MapLibre UI & Route Comparison Drawer
```

### Endpoints
- `POST /api/v1/routing/optimize`: Executes weather-aware route optimization. Returns both `base_route` and `weather_aware_route`, segment breakdown, weather provenance, and comparison metrics.
- `GET /api/v1/routing/defaults?vessel_type={type}`: Returns realistic standard naval dimensions, displacement, power, and fuel cost parameters for cargo, tanker, passenger, fishing, tug, service, or other vessels.

### Performance & Dataset Benchmarks
- **Dataset**: MARIS-Forecast NOAA Track A (`DOI: 10.5281/zenodo.21224009`, Hugging Face `mark000071/envship_v2_datasets`). Total targeted acquisition: 53.11 MB (48,000 train, 6,000 val, 6,000 test; vessel-disjoint across splits).
- **TFT Model**: 149,825 parameters.
  - Test MAE: **0.230 knots**
  - Test RMSE: **0.446 knots**
  - Test $R^2$: **0.9899**
  - CPU Inference Latency: **1.676 ms**
- **Operational Disclaimer**: All voyage durations, power values, and fuel metrics are explicit computational estimates based on documented engineering models. The system never claims certified fuel savings or guaranteed-safe routing.


---

## 🚢 Phase 2: Live Vessel Traffic Subsystem

The live vessel traffic subsystem connects server-side to the real AISStream network and provides real-time situational awareness on an interactive MapLibre GL map.

### Architectural Flow
```
AISStream (wss://stream.aisstream.io/v0/stream)
    │ (Server-side only; API key never sent to browser)
    ▼
AIS Ingestion Worker (app/workers/ais_worker.py)
    │ Normalization (PositionReport, ClassB, ShipStaticData)
    ▼
Redis Stream (stream:ais:raw)
    │ Consumer Group (group:ais:persistence)
    ▼
PostGIS Persistence Layer
    ├── vessel_identity (MMSI, IMO, name, ship type, dimensions)
    ├── current_vessel_state (latest coordinates, SOG, COG, heading, nav status; GiST indexed)
    └── vessel_position_history (spatial deadband: >100m, >5° heading, >180s)
    │
    ├── Redis Pub/Sub (channel:ais:broadcast)
    │       ▼
    │   FastAPI WebSocket Gateway (/ws/vessels)
    │       │ In-memory spatial viewport filtering
    │       ▼
    └── Frontend UI (MapLibre GL JS + Carto Dark Matter basemap)
```

### Endpoints
- `GET /api/v1/vessels`: Spatial bounding box query (`min_lat`, `min_lon`, `max_lat`, `max_lon`) returning GeoJSON FeatureCollection of live vessels.
- `GET /api/v1/vessels/{mmsi}`: Detailed vessel identity, latest kinematics, and recent historical track coordinates.
- `GET /api/v1/vessels/pipeline/status`: Real-time diagnostics of AIS connection state, messages received, persisted count, and DB vessel totals.
- `WS /ws/vessels`: Bi-directional WebSocket. Client transmits viewport bounding box (`{"type": "bbox", "bbox": [min_lat, min_lon, max_lat, max_lon]}`); server streams vessels within that viewport.

### Configuration
Configured via `.env`:
- `AISSTREAM_API_KEY`: AISStream API token (kept server-side).
- `AIS_BBOX_MIN_LAT`, `AIS_BBOX_MIN_LON`, `AIS_BBOX_MAX_LAT`, `AIS_BBOX_MAX_LON`: Initial bounding box (default: English Channel / Dover Strait `[49.5, -2.0, 51.5, 2.5]`).

---

## 🛰️ Phase 4: Dark-Vessel Intelligence Subsystem

The Dark-Vessel Intelligence subsystem provides spaceborne maritime radar surveillance by integrating Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) imagery with real AIS tracking on the single unified MapLibre canvas.

### Architectural Flow
```
Copernicus Data Space Ecosystem (CDSE)
   │ STAC API: https://stac.dataspace.copernicus.eu/v1/search
   ▼
Catalog Discovery & Asset Acquisition
   │ Level-1 Ground Range Detected (GRD) Interferometric Wide (IW) VV/VH
   ▼
Radiometric Preprocessing & Tiling
   │ Dual-channel stacking, log-dB stretch normalization, 640x640 sliding window tiles
   ▼
YOLO26s SAR Vessel Detector
   │ Pretrained on OpenSAR Insight SAR vessel dataset
   │ Bounding-box detection, confidence scoring & Non-Maximum Suppression (NMS)
   ▼
SAR Bilinear Georeferencing
   │ Projects image pixel space (x, y) to WGS84 (lon, lat) via scene corner tie-points
   ▼
PostGIS Spatiotemporal Correlation Engine
   │ Spatial threshold: 3.0 km radius
   │ Temporal window: ±15 minutes of SAR acquisition timestamp
   │ Queries PostGIS current_vessel_state & vessel_position_history
   ├── Matched: correlated with active AIS vessel  → AIS-MATCHED (Cyan ring)
   └── Unmatched: no correlating AIS broadcast   → AIS-UNMATCHED (Amber beacon)
   ▼
Unified MapLibre Frontend Interface
   │ Polygon scene footprint, interactive detection markers, AIS correlation inspector
```

### Endpoints
- `POST /api/v1/sar/search`: Discover available Sentinel-1 GRD scenes over a bounding box and acquisition date range via Copernicus STAC.
- `POST /api/v1/sar/jobs`: Asynchronously launch analysis job for a Sentinel-1 scene (download $\to$ preprocess $\to$ inference $\to$ correlate).
- `GET /api/v1/sar/jobs/{job_id}`: Track analysis job progress and stages (`QUEUED`, `DOWNLOADING`, `PREPROCESSING`, `INFERENCE`, `CORRELATING`, `COMPLETE`).
- `GET /api/v1/sar/scenes`: Retrieve previously analyzed SAR scenes stored in PostGIS.
- `GET /api/v1/sar/scenes/{scene_id}/detections`: Retrieve all vessel detections with full AIS correlation provenance for an analyzed scene.
- `GET /api/v1/sar/model/status`: Retrieve YOLO26s SAR model configuration, operational status, and held-out test metrics.

### Model Specifications & Evaluation
- **Architecture**: YOLO26s (Standard 2D object detection, single class `0 = vessel`)
- **Dataset**: OpenSAR Insight Sentinel-1 GRD filtered dataset (203 train, 13 val, 100 test images)
- **Training Environment**: Kaggle NVIDIA Tesla T4 GPU (FP16 mixed precision, patience 25 early stopping)
- **Training Progress**: 87 epochs completed, best epoch: 62
- **Held-Out Test Set Metrics**:
  - Precision: `0.4044` (40.44%)
  - Recall: `0.3152` (31.52%)
  - mAP50: `0.2335` (23.35%)
  - mAP50-95: `0.0471` (4.71%)
  - Measured local CPU inference latency: `57.10 ms` per 640x640 tile
- **Operational Status**: `TRAINED / OPERATIONAL` (Explicitly not certified or production-validated)
- **Generalization Gap**: Performance on the strictly held-out test set (100 images, 257 vessels) indicates an expected generalization gap relative to validation metrics. Real-world SAR returns exhibit variable sea-state clutter, speckle noise, and incidence angle dependencies.

### Provenance & Operational Ethics Disclaimer
- **Timestamped Snapshot Observations**: Synthetic Aperture Radar captures discrete observations at the exact satellite overpass epoch. Detections are labeled with their explicit acquisition timestamp and never characterized as "real-time radar" or "live SAR".
- **Strict Terminology**: Target contacts are classified strictly as `AIS-MATCHED` or `AIS-UNMATCHED`.
- **Zero Prejudicial Labeling**: An `AIS-UNMATCHED` contact indicates only that no AIS broadcast matched the target within the configured spatiotemporal tolerance ($\pm 15$ minutes, $3.0\text{ km}$). It must never be labeled as an "illegal vessel", "illegal ship", or "criminal vessel" (unmatched contacts may be small crafts without AIS carriage requirements, naval vessels, non-reporting crafts, or transient coverage gaps).
