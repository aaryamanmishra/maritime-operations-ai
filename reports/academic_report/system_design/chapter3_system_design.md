# Chapter 3: System Analysis and Design

## 3.1 System Requirements

### 3.1.1 Hardware Requirements
The Maritime Operations AI platform is designed for flexible deployment across local edge workstations, cloud virtual machines, and GPU-accelerated training environments.

#### Table 3.1: Hardware Infrastructure Requirements
| Component | Minimum Specification | Development Workstation | Cloud Training Environment (Kaggle/Colab) | Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **Processor (CPU)** | 4 Cores, x86_64 architecture | Intel Core i7-1185G7 (4 Cores, 8 Threads, up to 4.80 GHz) | Intel Xeon Dual-Core (2.20 GHz) | Native Sentinel-1 tiling, georeferencing, FastAPI concurrency |
| **System Memory (RAM)** | 16 GB DDR4 | 32 GB DDR4-3200 | 13.0 GB High-RAM | GeoTIFF band stacking and sliding-window matrix manipulation |
| **GPU Accelerator** | Optional (CPU fallback supported) | Intel Iris Xe Graphics (Shared) | NVIDIA Tesla T4 (16 GB GDDR6 VRAM, CUDA 12.2 / 13.0) | High-throughput YOLO training and batched model evaluation |
| **Storage Capacity** | 20 GB Solid State Drive | 1 TB NVMe SSD (PCIe 4.0) | 20 GB High-Speed Ephemeral Disk | Caching Sentinel-1 COG measurement bands and model weights |
| **Network Interface** | 10 Mbps Broadband | 1 Gbps Fiber Ethernet | 100 Mbps Dedicated Backbone | Copernicus CDSE scene streaming and live AISStream WebSocket |

---

### 3.1.2 Software Requirements
The software ecosystem leverages modern, high-performance, open-source technologies containerized via Docker for reproducibility:

#### Table 3.2: Software Environment, Frameworks, and Package Dependencies
| Software Layer | Technology / Package | Version | Operational Function in Maritime Operations AI |
| :--- | :--- | :--- | :--- |
| **Host Operating System** | Microsoft Windows 11 Pro / WSL2 | Build 22631 | Primary development host and Docker execution engine |
| **Containerization Engine** | Docker Desktop & Docker Compose | Engine 27.0+ | Microservice container orchestration (Backend, Worker, DB, Redis, UI) |
| **Backend Runtime** | Python | 3.12.15 | Asynchronous application runtime, ML pipeline, and geospatial math |
| **Web API Framework** | FastAPI | 0.115.0 | High-performance asynchronous RESTful API and WebSocket endpoints |
| **ASGI Server** | Uvicorn (Standard) | 0.31.0 | Asynchronous server with `uvloop` event loop and `httptools` parsing |
| **Spatial Database** | PostgreSQL + PostGIS | 16-3.4 (Alpine) | Relational persistence, spatial GIST indexing, `ST_DWithin` queries |
| **In-Memory Cache** | Redis | 7.0 (Alpine) | Geospatial coordinate caching (`GEOADD`), Pub/Sub, and TTL tracking |
| **Deep Learning Engine** | PyTorch | 2.2.0+ / 2.14.0 | Tensor calculations, automatic mixed precision (AMP), neural inference |
| **Object Detection Framework**| Ultralytics YOLO | 8.3.0+ | YOLO11n, YOLO11s, and YOLOv8s ship detection architectures |
| **Frontend Framework** | React + TypeScript | 19.0.0 / 5.7.2 | Component-driven user interface and strict static type safety |
| **Build Tooling** | Vite | 6.1.0 | Fast development hot-module reloading and optimized production bundling |
| **Geospatial Map Renderer** | MapLibre GL | 6.11.2 | Hardware-accelerated WebGL vector map rendering and radar overlays |
| **Scientific Computation** | NumPy & Pandas | 2.4.6 / 3.0.6 | Matrix percentile clipping, normalization, and evaluation metric audits |

---

## 3.2 System Architecture

The overall system architecture follows a decoupled, cloud-native microservices topology structured into five distinct operational tiers:

```
[Copernicus Sentinel-1 SAR]       [AISStream WebSocket Stream]
            |                                  |
            v                                  v
+-----------------------+          +-----------------------+
|  Acquisition Client   |          |  AIS Ingestion Worker |
+-----------------------+          +-----------------------+
            |                                  |
            v                                  v
+-----------------------+          +-----------------------+
| Preprocessor & Tiler  |          | PostGIS & Redis Cache |
+-----------------------+          +-----------------------+
            |                                  |
            v                                  |
+-----------------------+                      |
|  YOLO Vessel Detector |                      |
+-----------------------+                      |
            |                                  |
            v                                  |
+-----------------------+                      |
|   GCP Georeferencer   |                      |
+-----------------------+                      |
            |                                  |
            +-----------------+----------------+
                              |
                              v
              +-------------------------------+
              | Kinematic Correlation Engine  |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              | Tactical React / MapLibre UI  |
              +-------------------------------+
```

*Refer to Figure 3.1: Overall System Architecture (`figures/fig3_1_overall_system_architecture.svg` / `.mmd`).*

### Architectural Tiers:
1. **External Ingestion Tier:** Interacts with external data providers over secure TLS channels:
   - **Copernicus CDSE Client:** Interrogates the Copernicus Data Space Ecosystem STAC and OData APIs to download Sentinel-1 Level-1 GRD swaths.
   - **AISStream Client:** Ingests live global AIVDM/AIVDO vessel broadcast messages via persistent WebSocket connections.
2. **Asynchronous Background Processing Tier:** Executes high-throughput, non-blocking computational pipelines:
   - **AIS Ingestion Worker:** Filters, validates, batches, and stores streaming AIS records in PostgreSQL and Redis.
   - **SAR Analysis Service:** Manages background execution jobs for multi-spectral band extraction, dynamic radiometric calibration, and sliding-window tiling.
3. **Machine Learning & Georeferencing Inference Tier:**
   - **YOLO Detector Engine:** Executes batched tensor inference on native $640 \times 640$ chips using sensor-matched weights (`yolo26s_sar_vessel.pt` for GRD; `yolo26s_slc_vessel.pt` for opt-in SLC).
   - **Non-Maximum Suppression & Deduplication:** Resolves intra-tile overlaps and cross-tile duplicates across adjacent sliding windows.
   - **GCP Georeferencer:** Projects target pixel centroids $(X, Y)$ into WGS84 geographic coordinates $(\text{Lon}, \text{Lat})$ via piecewise bilinear interpolation over product Geolocation Grid Points.
4. **Geospatial Persistence & Correlation Tier:**
   - **PostgreSQL 16 with PostGIS 3.4:** Persists satellite scenes, vessel identities, historical vessel trajectories, localized radar detections, and correlation records with spatial GIST indices.
   - **Redis 7 In-Memory Cache:** Maintains ephemeral spatial indices (`GEOADD`) for active vessels and broadcasts live position deltas over Redis Pub/Sub channels.
   - **Kinematic Correlation Engine:** Performs spatial candidate retrieval ($\pm 15\,\text{min}$, $4.5\,\text{km}$), computes dead reckoning projections, constructs dynamic 3-component uncertainty budgets, and evaluates soft-edge Gaussian decision boundaries.
5. **Tactical Presentation Tier:**
   - **FastAPI Core Gateway:** Exposes RESTful JSON endpoints, GeoJSON layer generators, and WebSocket traffic streams.
   - **React 19 / MapLibre GL Dashboard:** Renders interactive vector charts, high-contrast SAR swath overlays, dark-vessel alerts, and detailed provenance dossiers for maritime watch officers.

---

## 3.3 Use-Case Analysis

*Refer to Figure 3.11: UML Use-Case Diagram (`figures/fig3_11_uml_use_case_diagram.svg` / `.mmd`).*

The primary operational use cases supported by the platform are:
- **UC-01: Monitor Real-Time Vessel Traffic:** Watch officers inspect real-time vessel movements, filter vessels by maritime classification (cargo, tanker, fishing), and view historical kinematic trajectories.
- **UC-02: Query Satellite SAR Passes:** Operators select an Area of Interest (bounding box) and date range. The system queries the Copernicus STAC catalog and visualizes satellite pass footprints.
- **UC-03: Execute Native-Resolution Ship Detection:** Users trigger asynchronous satellite analysis jobs on selected scenes. The engine tiles the swath at native resolution, runs the YOLO detector, deduplicates twin candidates, and georeferences targets.
- **UC-04: Classify Dark-Vessel Candidates:** The correlation engine compares radar detections against historical AIS records, automatically categorizing targets into `AIS-MATCHED`, `AMBIGUOUS`, `AIS-UNMATCHED` (potential dark vessel), and `INSUFFICIENT-COVERAGE`.
- **UC-05: Inspect Positional Uncertainty Budget & Provenance:** Tactical intelligence analysts open a detection dossier to inspect calculated dead-reckoning movement, time delta, $z$-score, Gaussian likelihood score, and uncertainty budget breakdown ($\sigma_{sar}, \sigma_{ais}, \sigma_{motion}$).
- **UC-06: Plan Weather-Optimized Voyage Route:** Navigators input origin and destination coordinates. The routing engine utilizes a Temporal Fusion Transformer (TFT) surrogate and Kwon wave-degradation model to generate minimum-fuel, weather-safe passage tracks.
- **UC-07: Ingest & Cleanse AIS Broadcast Stream:** The background ingestion worker continuously decodes AIVDM messages, validates kinematic sanity (SOG $\le 70\text{ kn}$, COG $\in [0, 360^\circ)$), and updates spatial databases.
- **UC-08: Ingest Copernicus Swath GeoTIFFs:** The acquisition adapter authenticates via CDSE OData, validates local caching, streams missing measurement bands, and extracts product metadata.
- **UC-09: Monitor System Health & Subsystems:** Administrators evaluate Kubernetes-compatible `/api/health/live` and `/api/health/ready` endpoints to monitor database connection pools and cache health.

---

## 3.4 Database Relational Schema & ER Diagram

*Refer to Figure 3.10: Database Entity-Relationship Diagram (`figures/fig3_10_database_er_diagram.svg` / `.mmd`).*

The spatial database schema is implemented in PostgreSQL 16 with the PostGIS 3.4 extension. All geometries are strictly modeled in canonical WGS84 (EPSG:4326).

```
+--------------------------------------------------------------------------------------------------+
|                               POSTGIS RELATIONAL DATABASE SCHEMA                                 |
+--------------------------+----------------------------+------------------------------------------+
| Table Name               | Primary Key                | Core Foreign Keys & Spatial Columns      |
+--------------------------+----------------------------+------------------------------------------+
| sar_scene                | scene_id VARCHAR(128)      | footprint GEOMETRY(POLYGON, 4326) [GIST] |
| sar_detection            | id BIGINT (Autoincrement)  | scene_id FK -> sar_scene.scene_id        |
|                          |                            | geom GEOMETRY(POINT, 4326) [GIST]        |
| ais_sar_correlation      | id BIGINT (Autoincrement)  | sar_detection_id FK (Unique)             |
|                          |                            | candidate_mmsi FK -> vessel_identity     |
| vessel_identity          | mmsi BIGINT                | None (Root Vessel Dimension Table)       |
| current_vessel_state     | mmsi BIGINT (PK, FK)       | mmsi FK -> vessel_identity.mmsi          |
|                          |                            | geom GEOMETRY(POINT, 4326) [GIST]        |
| vessel_position_history  | id BIGINT (Autoincrement)  | mmsi FK -> vessel_identity.mmsi          |
|                          |                            | geom GEOMETRY(POINT, 4326) [GIST]        |
+--------------------------+----------------------------+------------------------------------------+
```

### Table Definitions and Constraints:
1. **`sar_scene` Table:**
   - `scene_id` (VARCHAR 128, PK): Official Copernicus product identifier (e.g., `S1C_IW_GRDH_1SDV_20261026T062251_...`).
   - `acquisition_time` (TIMESTAMPTZ, indexed): Satellite center overpass epoch in UTC.
   - `platform` (VARCHAR 64): Satellite identifier (`Sentinel-1A` or `Sentinel-1C`).
   - `orbit_pass` (VARCHAR 32): Orbital direction (`ASCENDING` or `DESCENDING`).
   - `footprint` (GEOMETRY(POLYGON, 4326), GIST indexed): Full geographic boundary polygon of the radar swath.
   - `processing_level` (VARCHAR 32): `LEVEL1_GRD` or `LEVEL1_SLC`.
   - `polarization` (VARCHAR 32): Microwave channel (`VV+VH`).
   - `scene_metadata` (JSONB): Technical raster dimensions, GCP count, and orbit details.
2. **`sar_detection` Table:**
   - `id` (BIGINT, PK): Surrogate primary key.
   - `detection_id` (VARCHAR 64, Unique, indexed): UUID string identifier.
   - `scene_id` (VARCHAR 128, FK, on delete CASCADE): Reference to parent `sar_scene`.
   - `geom` (GEOMETRY(POINT, 4326), GIST indexed): Target centroid geographic point.
   - `latitude` / `longitude` (NUMERIC(9,6)): Canonical WGS84 coordinates in decimal degrees.
   - `pixel_bbox` (JSONB): Bounding box in parent-scene pixel coordinates `{"xmin": 1200, "ymin": 3400, "xmax": 1245, "ymax": 3430}`.
   - `confidence` (NUMERIC(4,3)): Deep-learning detection probability score $[0.0, 1.0]$.
   - `model_version` (VARCHAR 64): Identity of the model generating the detection.
   - `length_m` / `heading_deg` (NUMERIC): Estimated physical vessel dimensions.
3. **`ais_sar_correlation` Table:**
   - `id` (BIGINT, PK): Surrogate primary key.
   - `sar_detection_id` (BIGINT, FK, Unique): One-to-one foreign key linking `sar_detection`.
   - `candidate_mmsi` (BIGINT, FK, indexed): Correlated MMSI identifier (nullable).
   - `time_difference_seconds` (NUMERIC(8,2)): Time delta $\Delta t = T_{sat} - T_{ais}$.
   - `distance_km` (NUMERIC(8,3)): Kinematic dead-reckoned geodesic distance.
   - `match_status` (VARCHAR 32, indexed): `AIS-MATCHED`, `AMBIGUOUS`, `AIS-UNMATCHED`, or `INSUFFICIENT-COVERAGE`.
   - `matching_method` (VARCHAR 64): Algorithm identifier (`uncertainty-kinematic-geodesic-v2`).
   - `provenance` (JSONB): Complete audit record containing $z$-score, likelihood score, hard gate, soft gate, and uncertainty budget components.
4. **`vessel_identity` Table:**
   - `mmsi` (BIGINT, PK): Maritime Mobile Service Identity (unique 9-digit transponder ID).
   - `imo` (INTEGER, nullable): International Maritime Organization registration number.
   - `name` (VARCHAR 128): Vessel commercial name.
   - `ship_type` (INTEGER): IMO vessel classification code (Cargo, Tanker, Fishing, etc.).
   - `length` / `beam` / `draught` (NUMERIC): Physical vessel dimensions in meters.
5. **`current_vessel_state` Table:**
   - `mmsi` (BIGINT, PK, FK): Reference to `vessel_identity`.
   - `timestamp` (TIMESTAMPTZ, indexed): Latest broadcast timestamp in UTC.
   - `geom` (GEOMETRY(POINT, 4326), GIST indexed): Latest position coordinate.
   - `sog` (NUMERIC(5,2)): Speed Over Ground in knots.
   - `cog` (NUMERIC(5,2)): Course Over Ground in degrees $[0.0, 360.0)$.
6. **`vessel_position_history` Table:**
   - `id` (BIGINT, PK): Surrogate primary key.
   - `mmsi` (BIGINT, FK, indexed): Reference to `vessel_identity`.
   - `timestamp` (TIMESTAMPTZ, indexed): Message broadcast epoch in UTC.
   - `geom` (GEOMETRY(POINT, 4326), GIST indexed): Historical coordinate point.
   - `sog` / `cog` / `heading` (NUMERIC): Kinematic telemetry at broadcast time.

---

## 3.5 Core Module Descriptions

*Refer to Table 3.3 (`tables/table3_3_project_modules.csv`) for the complete project module inventory.*

### Module 1: Sentinel-1 Acquisition & CDSE Catalog Adapter
- **Location:** `backend/app/infrastructure/adapters/copernicus/`
- **Classes:** `CopernicusCatalogClient`, `CopernicusAcquisitionClient`
- **Functionality:** Authenticates against Copernicus Keycloak using OAuth2 client credentials. Queries the STAC API for Sentinel-1 products intersecting the target bounding box and temporal interval. Downloads product SAFE zip archives via OData stream, extracting measurement bands (`measurement_vv.tiff`, `measurement_vh.tiff`) and annotation XML files into local storage (`backend/data/sar_cache/`).

### Module 2: SAR Radiometric Preprocessing & Native Tiling Engine
- **Location:** `backend/app/infrastructure/adapters/copernicus/`
- **Classes:** `SARPreprocessor`, `SLCSARPreprocessor`, `SARTile`
- **Functionality:** Ingests 16-bit raw digital number ($DN$) GeoTIFF rasters. Implements dynamic $2.0\%$ to $99.5\%$ percentile clipping to eliminate high-intensity speckle outliers while preserving vessel point scatterers. Normalizes bands to 8-bit $[0, 255]$ arrays and synthesizes a 3-channel pseudo-RGB stack:
  $$\text{Ch}_0 = \text{VV}_{norm}, \quad \text{Ch}_1 = \text{VH}_{norm}, \quad \text{Ch}_2 = \frac{\text{VV}_{norm} + \text{VH}_{norm}}{2}$$
  Executes sliding-window slicing with a window size of $640 \times 640$ pixels and a stride of $576\text{ px}$ ($64\text{ px}$ overlap), zero-padding border tiles and recording local-to-global coordinate offsets $(x_{offset}, y_{offset})$.

### Module 3: Deep-Learning SAR Ship Detection & Deduplication Engine
- **Location:** `backend/app/domain/sar_intelligence/`
- **Classes:** `YOLO26sVesselDetector`, `non_max_suppression_boxes`, `deduplicate_scene_detections`
- **Functionality:** Loads sensor-matched PyTorch `DetectionModel` weights. Executes batched forward passes with confidence threshold $0.25$. Performs intra-tile Non-Maximum Suppression ($IoU \ge 0.35$). Maps bounding boxes back to parent scene pixel space:
  $$X = x_{chip} + x_{offset}, \quad Y = y_{chip} + y_{offset}$$
  Executes conservative cross-tile deduplication across overlapping tile boundaries ($IoU \ge 0.30$ or near-concentric enclosure $IoMin \ge 0.50$), retaining the primary high-confidence target while embedding twin provenance records.

### Module 4: GCP-Based Piecewise Bilinear Georeferencer
- **Location:** `backend/app/infrastructure/adapters/copernicus/`
- **Classes:** `SARGeoreferencer`, `GeoPoint`, `PixelPoint`
- **Functionality:** Parses the grid of Geolocation Grid Points (GCPs) embedded in the product annotation XML. Performs bisection search along image lines to locate enclosing GCP rows $[L_0, L_1]$, followed by bisection search along range columns to find enclosing pixels $[P_0, P_1]$. Calculates normalized interpolation weights:
  $$u = \frac{X - P_0}{P_1 - P_0}, \quad v = \frac{Y - L_0}{L_1 - L_0}$$
  Applies 2D bilinear interpolation over GCP coordinates to compute WGS84 $(\text{Lon}, \text{Lat})$. Validates geometric invariants (finite real numbers, latitude $\in [-90, 90]$, antimeridian normalization $\text{Lon} \in [-180, 180)$).

### Module 5: Continuous Real-Time AIS Ingestion & Spatial Cache Worker
- **Location:** `backend/app/workers/`
- **Classes:** `AISStreamClient`, `AIVDMParser`, `VesselRepository`
- **Functionality:** Establishes an asynchronous WebSocket connection to AISStream.io. Decodes dynamic position reports (Message Types 1, 2, 3) and static voyage reports (Message Type 5). Validates kinematic limits ($0 \le SOG \le 70\text{ kn}$, $0 \le COG < 360^\circ$). Performs atomic batched upserts into PostgreSQL `current_vessel_state` and appends historical tracks. Simultaneously populates Redis geospatial keys (`GEOADD live_vessels lon lat mmsi`) with a 1-hour expiration TTL.

### Module 6: Kinematic Uncertainty-Aware Spatiotemporal Correlation Engine
- **Location:** `backend/app/domain/sar_intelligence/`
- **Classes:** `AISSARCorrelationEngine`
- **Functionality:** Formulates candidate queries over a temporal window $[T_{sat} - 15\,\text{min}, T_{sat} + 15\,\text{min}]$ and spatial radius of $4.5\,\text{km}$. Extrapolates candidate positions via dead reckoning using SOG and COG. Calculates geodesic Haversine distance $dist_m$. Synthesizes a composite 3-component uncertainty budget:
  $$\sigma_{total} = \sqrt{\sigma_{sar}^2 + \sigma_{ais}^2 + \sigma_{motion}^2}$$
  Calculates normalized standard score $z = \frac{dist_m}{\sigma_{total}}$ and Gaussian likelihood score $S = \exp(-0.5 \cdot z^2)$. Evaluates hard gates ($dist_m \le 3000\,\text{m}$, $z \le 2.5$) and soft gates ($z \le 4.0$). Conducts a $35\,\text{km}$ regional sector coverage check when no candidate matches to distinguish dark vessels from sensor coverage blindspots.

### Module 7: Tactical Geospatial Command Dashboard
- **Location:** `frontend/src/components/`
- **Components:** `MaritimeMap.tsx`, `DarkVesselControlPanel.tsx`, `HealthStatus.tsx`
- **Functionality:** Provides an interactive MapLibre GL map interface with high-contrast SAR swath boundaries, real-time AIS traffic markers, and color-coded detection contacts (Green = `AIS-MATCHED`, Amber = `AMBIGUOUS`, Red = `AIS-UNMATCHED` / Dark Vessel, Gray = `INSUFFICIENT-COVERAGE`). Enables one-click scene acquisition, status polling, and target dossier inspection.

### Module 8: Weather Routing & Neural Vessel Performance Engine
- **Location:** `backend/app/domain/weather_routing/`, `backend/app/domain/ml_performance/`
- **Classes:** `WeatherRoutingService`, `VesselPerformanceTFT`, `TFTPerformancePredictor`
- **Functionality:** Implements a Temporal Fusion Transformer (TFT) deep neural surrogate trained on NOAA/MARIS vessel telemetry to predict actual Speed Over Ground degradation under wave height ($H_s$), wave direction, and wind stress. Couples predictions with Kwon empirical wave loss and the Searoute graph engine to synthesize weather-optimal, minimum-fuel passage tracks.

---

## 3.6 API Architecture

*Refer to Table 3.4 (`tables/table3_4_api_endpoints.csv`) for the complete RESTful endpoint specification.*

The backend exposes an asynchronous RESTful API structured under `/api/v1/`:
- **`POST /api/v1/sar/search`:** Receives spatial bounding box $[minLon, minLat, maxLon, maxLat]$, date boundaries, and modality (`GRD` or `SLC`). Queries Copernicus STAC and returns `List[SARSceneSummary]`.
- **`POST /api/v1/sar/jobs`:** Initiates asynchronous background processing for a selected scene, returning HTTP 202 Accepted with a unique `job_id`.
- **`GET /api/v1/sar/jobs/{job_id}`:** Returns live execution status (`PENDING`, `ACQUIRING`, `PREPROCESSING`, `INFERRING`, `CORRELATING`, `COMPLETED`, `FAILED`).
- **`GET /api/v1/sar/scenes/{scene_id}/detections`:** Returns all detected vessels for a scene with bounding boxes, confidence, WGS84 coordinates, and AIS correlation dossiers.
- **`GET /api/v1/vessels/live`:** Streams live vessel coordinates from Redis geospatial cache formatted as GeoJSON FeatureCollections.
- **`POST /api/v1/routing/optimize`:** Computes weather-optimal maritime routes with predicted speed loss and fuel savings.
- **`GET /api/health/live` & `/api/health/ready`:** Standard probes verifying container health, PostGIS connection pooling, and Redis response latency.

---

## 3.7 End-to-End Data Flow

*Refer to Figure 3.2: End-to-End Data Flow Diagram (`figures/fig3_2_end_to_end_data_flow.svg` / `.mmd`).*

The operational workflow from satellite trigger to dark-vessel alert proceeds through eleven discrete phases:
1. **Continuous AIS Ingestion:** The background worker ingests streaming AIS messages, updating PostGIS `current_vessel_state` and Redis `live_vessels` at $\sim 100\text{ messages/second}$.
2. **Tactical BBox Selection:** The watch officer defines an Area of Interest on the MapLibre GL map.
3. **STAC Scene Discovery:** The frontend queries `POST /api/v1/sar/search`. The backend queries Copernicus CDSE and renders available satellite footprints.
4. **Job Initiation:** The operator selects an overpass and clicks "Analyze Scene." The frontend issues `POST /api/v1/sar/jobs` and begins background status polling.
5. **Asset Acquisition:** The acquisition client checks local cache (`sar_cache/`); if missing, it downloads measurement GeoTIFFs via CDSE OData streams.
6. **Radiometric Normalization & Native Tiling:** The preprocessor performs $2.0\%\text{--}99.5\%$ percentile clipping and slices the swath into overlapping $640 \times 640$ chips.
7. **Batched Neural Inference:** The YOLO detector performs forward passes across all chips, applying confidence thresholding ($\ge 0.25$) and intra-tile NMS.
8. **Sub-Pixel GCP Georeferencing:** Target centroids are projected through the product GCP grid via piecewise bilinear interpolation to generate WGS84 coordinates.
9. **Cross-Tile Deduplication:** Overlapping candidate pairs are evaluated; duplicates are consolidated with provenance links.
10. **Spatiotemporal Kinematic Correlation:** PostGIS queries historical AIS pings within $\pm 15\,\text{min}$ and $4.5\,\text{km}$. The engine projects candidate positions via dead reckoning, computes composite uncertainty budgets, and evaluates decision boundaries.
11. **Tactical Presentation & Alerting:** Detections are persisted in PostGIS. The frontend completes polling, loads GeoJSON layers, and displays color-coded markers and dossiers on the map.

---

## 3.8 Security, Operational Integrity, and Ethical Guardrails

1. **Secret Isolation:** API credentials (`CDSE_USERNAME`, `CDSE_PASSWORD`, `AISSTREAM_API_KEY`) and database passwords are strictly isolated within environment files (`.env`), injected into container runtimes, and sanitized from logs and API payloads.
2. **Defamation & Legal Integrity Guardrails:** The platform strictly enforces ethical naming taxonomy. Unmatched targets are labeled as `AIS-UNMATCHED` or "Potential Dark-Vessel Candidate." The system is programmatically prohibited from generating assertions of "illegal", "pirate", or "criminal" vessels, maintaining adherence to maritime legal standards.
3. **Sensor Coverage Verification:** Before designating an unmatched contact as a potential dark vessel, the engine executes a $35\text{ km}$ regional audit query. If zero AIS vessels are logged across the entire sector, the contact is categorized as `INSUFFICIENT-COVERAGE`, preventing false alerts caused by terrestrial receiver terrain shadowing or satellite AIS outages.
