# Maritime Operations AI: Revised System Architecture Specification

## 1. Executive Summary & Revised Architectural Tenets

Maritime Operations AI consolidates three core capabilities into a single unified geospatial map platform:
1. **Live Vessel Traffic**: Scalable AIS ingestion, current state vs. historical trajectory separation, configurable retention, and decoupled WebSocket viewport filtering.
2. **AI Weather-Aware Routing**: A two-tier optimization system where a deep-learning vessel performance surrogate (candidate: Temporal Fusion Transformer / temporal sequence model) predicts environmental speed degradation/power penalty from AIS history, vessel dimensions, and weather context, which feeds a naval architecture energy model over a base SeaRoute maritime network.
3. **Dark-Vessel Intelligence**: Event-driven, on-demand Copernicus Sentinel-1 STAC discovery and scene processing; candidate lightweight SAR ship detector chosen after rigorous dataset/license audit; and PostGIS spatiotemporal correlation surfacing non-broadcasting contacts without prejudicial labeling.

### Revised Core Principles
- **Deep Learning as a First-Class Citizen in Routing**: Rather than naive pure A* or pure physics, deep learning predicts vessel speed loss / power responsiveness given historical operational profiles and dynamic sea states.
- **Strict Ingestion-Viewport Decoupling**: AIS ingestion is a fixed, non-viewport-dependent background process feeding Redis Streams and PostGIS. Frontend viewport filtering operates strictly at the WebSocket egress layer.
- **Demand-Driven / Cache-Driven Weather**: Weather forecast grids are fetched from Open-Meteo strictly on-demand for active routing corridors and cached with explicit TTLs in Redis / PostGIS.
- **Event-Driven SAR Pipeline**: Large Sentinel-1 GRD asset downloads occur only upon explicit user request / AOI selection after STAC catalogue discovery.
- **Scalable AIS Data Lifecycle**: Hard separation between `current_vessel_state` (fast in-place upsert) and `vessel_position_history` (thinned, partitioned, configurable retention).
- **Zero Out-of-Pocket Student Development Target**: 100% locally self-hosted via Docker Compose without requiring paid services, proprietary SDKs, or active cloud credit cards.

---

## 2. Updated Component Diagram

```mermaid
flowchart TB
    subgraph Client["Presentation Layer (Client Browser)"]
        UI["React 19 / TypeScript / Vite"]
        MapEngine["MapLibre GL JS + Deck.gl"]
        WSClient["WebSocket Client (Auto-Reconnect / Viewport BBox)"]
        UI --> MapEngine
        UI --> WSClient
    end

    subgraph Edge["Reverse Proxy Gateway"]
        Nginx["Nginx / Caddy Proxy (Port 80 / 443)"]
    end

    subgraph BackendMonolith["Backend Application Runtime (FastAPI Modular Monolith)"]
        APIGateway["FastAPI HTTP Router"]
        WSGateway["WebSocket Gateway<br/>(In-Memory Viewport Spatial Filtering)"]
        
        subgraph DomainModules["Domain Modules"]
            Vessels["domain.vessel_traffic<br/>(Kinematics, Thinning, State Management)"]
            MLPerf["domain.ml_performance<br/>(DL Vessel Performance & Speed Degradation Engine)"]
            Routing["domain.weather_routing<br/>(SeaRoute Network + Hydrodynamics + A* Optimization)"]
            SAR["domain.sar_intelligence<br/>(Event-Driven STAC Fetcher, ML Detector, Correlator)"]
        end
        
        subgraph Adapters["Infrastructure Adapters (Circuit Breakers / Retries)"]
            AISAdapter["AISStream Adapter"]
            WeatherAdapter["Open-Meteo Marine On-Demand Adapter"]
            CopernicusAdapter["Copernicus CDSE STAC / OData Adapter"]
        end
        
        subgraph Repositories["Data Access Layer (SQLAlchemy 2.0 Async)"]
            VesselRepo["Vessel State & History Repo"]
            WeatherRepo["Weather Cache Repo"]
            RouteRepo["Route & Voyage Repo"]
            SARRepo["SAR Scene & Detection Repo"]
            MLRepo["Model Metadata & Registry Repo"]
        end
        
        APIGateway --> DomainModules
        WSGateway --> DomainModules
        DomainModules --> Repositories
        DomainModules --> Adapters
        Routing --> MLPerf
    end

    subgraph Workers["Asynchronous Background Workers"]
        AISWorker["AIS Ingestion Worker<br/>(Independent Feed -> Redis Stream)"]
        AISPersistWorker["AIS Persistence Worker<br/>(Redis Consumer -> Micro-batch State/History)"]
        SARWorker["SAR Task Worker<br/>(On-Demand Scene Acquisition & Tiling)"]
    end

    subgraph Storage["Persistence & Messaging Tier"]
        RedisPubSub[("Redis 7<br/>- stream:ais:raw<br/>- channel:ais:broadcast<br/>- weather:cache (Primary short-lived TTL cache)")]
        PostgresPostGIS[("PostgreSQL 16 + PostGIS 3.4<br/>- vessel_identity<br/>- current_vessel_state (Target benchmark: &lt;50ms viewport queries)<br/>- vessel_position_history (Partitioned)<br/>- route_requests & route_results<br/>- sar_scenes & sar_detections<br/>- ml_models")]
    end

    %% Client communication
    Client <-->|HTTPS / REST API| Nginx
    Client <-->|WSS / Viewport Stream| Nginx
    Nginx --> APIGateway
    Nginx --> WSGateway

    %% Decoupled Ingestion Flow
    AISAdapter <-->|Continuous TLS WSS| AISWorker
    AISWorker -->|XADD stream:ais:raw| RedisPubSub
    RedisPubSub -->|XREADGROUP Consumer| AISPersistWorker
    AISPersistWorker -->|In-Place Upsert| VesselRepo
    AISPersistWorker -->|Thinned Insert| VesselRepo
    AISPersistWorker -->|PUBLISH channel:ais:broadcast| RedisPubSub
    RedisPubSub -->|Cross-Process Pub/Sub| WSGateway

    %% Demand-Driven Weather Flow
    Routing -->|Check Cache / Fetch Corridor| WeatherAdapter
    WeatherAdapter <-->|On-Demand REST (TTL Cached)| RedisPubSub
    WeatherAdapter --> WeatherRepo

    %% Event-Driven SAR Flow
    SAR -->|STAC Scene Search| CopernicusAdapter
    SARWorker -->|On-Demand Asset Fetch & Inference| SARRepo

    %% Persistence
    VesselRepo --> PostgresPostGIS
    RouteRepo --> PostgresPostGIS
    SARRepo --> PostgresPostGIS
    MLRepo --> PostgresPostGIS
```

---

## 3. Decoupled Ingestion & Real-Time Flow

### 3.1 AIS Ingestion Pipeline
1. `AISStream` connects strictly through a background ingestion process (`ais_ingestion_worker`).
2. The ingestion worker does NOT know about or respond to client browser viewports. It listens to configured regional maritime zones (or global stream where capacity allows).
3. Ingested messages are validated, normalized to canonical schemas (`NormalizedVesselEvent`), and written to Redis Stream `stream:ais:raw`.
4. An independent consumer (`ais_persistence_worker`) processes stream micro-batches using consumer group `group:ais:persistence`:
   - **`vessel_identity`**: Tracks static vessel metadata (`mmsi`, `ship_name`, `imo`, `call_sign`, `ship_type`, `dimension_to_bow`, `dimension_to_stern`, `dimension_to_port`, `dimension_to_starboard`, `length_m`, `beam_m`).
   - **`current_vessel_state`**: Updates the latest coordinate, SOG, COG, heading, and status using `INSERT ... ON CONFLICT (mmsi) DO UPDATE` with GiST indexing on `geom` (target benchmark: &lt;50ms viewport queries).
   - **`vessel_position_history`**: Subject to deadband / spatial thinning filter: inserts only if position moved $> 100\text{ m}$ (Haversine formula), or heading changed $> 5^\circ$ (with $360^\circ$ angular wraparound), or elapsed time $> 180\text{ s}$ since last stored track point.
   - **Realtime Fanout**: Publishes the kinematic update to Redis Pub/Sub `channel:ais:broadcast`.
5. **WebSocket Gateway**: Connected client browser instances maintain a WebSocket session with `/ws/vessels`, transmitting their current map bounding box. The WebSocket gateway filters Redis Pub/Sub events in memory against active client bounding boxes before sending to the client, decoupling client viewports completely from the external AISStream connection.

### 3.2 Demand-Driven Weather Ingestion Pipeline
1. User requests a voyage route from Origin to Destination with departure time $T_0$.
2. The routing engine generates candidate navigation corridor waypoints from the SeaRoute network.
3. System checks Redis `weather:cache:{h3_index_or_grid_cell}:{timestamp_hour}`.
4. If cached and valid within TTL (e.g., 6 hours), data is returned instantly.
5. If missing or stale, `weather_adapter` issues a focused batch query to Open-Meteo Marine API covering only the required route bounding box / points.
6. Retrieved grids are stored in Redis as the primary short-lived cache with TTL. The cache is not duplicated into PostGIS unless a concrete persistence/provenance requirement later justifies it.

### 3.3 Event-Driven SAR Intelligence Pipeline
1. **Scene Discovery**: User selects an Area of Interest (AOI) bounding box on the map and an acquisition temporal window.
2. The API queries Copernicus Data Space Ecosystem (CDSE) STAC catalogue metadata (`https://stac.dataspace.copernicus.eu/v1/search`, collection `sentinel-1-grd`).
3. The UI presents available Sentinel-1 GRD scene footprints, acquisition timestamps, and pass directions on the unified map.
4. **On-Demand Processing**: User explicitly triggers analysis for a chosen scene.
5. The `SARAnalysisService` orchestrates stateful job progression (`QUEUED` $\to$ `DOWNLOADING` $\to$ `PREPROCESSING` $\to$ `INFERENCE` $\to$ `CORRELATING` $\to$ `COMPLETE`):
   - **Asset Acquisition**: Discovers direct GeoTIFF / quicklook assets or acquires raw assets via Copernicus Keycloak OAuth2 token authentication.
   - **Radiometric Preprocessing**: Stacks dual-polarization channels (VV + VH), applies logarithmic decibel normalization ($10 \cdot \log_{10}(\sigma_0)$) scaled to $[0, 255]$, and slices large swaths into $640 \times 640$ overlapping tiles (64px overlap stride).
   - **YOLO26s Inference**: Executes the trained single YOLO26s SAR vessel detector across all tiles; applies Non-Maximum Suppression (NMS, IoU 0.4) to eliminate edge duplication.
   - **Bilinear Georeferencing**: Projects pixel coordinates $(x, y)$ to WGS84 geographic coordinates $(\text{lon}, \text{lat})$ via scene corner coordinate tie-points.
   - **PostGIS Spatiotemporal Correlation**: Spatially matches detections against recorded AIS targets (`current_vessel_state` and `vessel_position_history`) within $\pm 15$ minutes and $3.0\text{ km}$ distance threshold using `ST_DWithin`.
   - **Persistence**: Persists `sar_scenes`, `sar_detections`, and `ais_sar_correlations` in PostGIS.
6. The map renders the discrete scene observation footprint with matched vs. unmatched detections. Satellite passes are explicitly presented as timestamped snapshot observations.

---

## 4. SAR Detection & Spatiotemporal Correlation Engine

### 4.1 Single Deep Learning Model: YOLO26s
- **Model**: `YOLO26s` architecture from Ultralytics / OpenSAR Insight (`task: detect`, `nc: 1`, class 0 = vessel).
- **Dataset**: OpenSAR Insight Sentinel-1 GRD filtered dataset (203 train, 13 val, 100 test images; 640x640 resolution).
- **Training Environment**: Kaggle NVIDIA Tesla T4 GPU with FP16 Automatic Mixed Precision (`amp=True`).
- **Training Progress**: 87 epochs completed (Best epoch: 62, early stopping patience 25).
- **Held-Out Test Set Metrics**: Precision: `0.4044`, Recall: `0.3152`, mAP50: `0.2335`, mAP50-95: `0.0471`.
- **Measured Local CPU Inference Latency**: `57.10 ms` per 640x640 tile.
- **Operational Status**: `TRAINED / OPERATIONAL` (Explicitly not certified or production-validated).
- **Generalization Gap**: Performance on the strictly held-out test split (0.2335 mAP50 vs. ~0.515 val mAP50) reflects the inherent challenge of complex coastal radar backscatter, speckle noise, and uncalibrated sea-clutter states. Detections are operational probability alerts rather than certified navigational ground truth.
- **Format**: PyTorch checkpoint (`yolo26s_sar_vessel.pt`) with standard weights metadata JSON (`yolo26s_metadata.json`).


### 4.2 PostGIS Correlation Rules & Explainability
For each detected radar contact $D$ at timestamp $T_{SAR}$ and position $(\text{lat}_D, \text{lon}_D)$:
```sql
SELECT mmsi, latitude, longitude, timestamp,
       ST_Distance(
           geom::geography,
           ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
       ) AS distance_meters
FROM (
    SELECT mmsi, latitude, longitude, timestamp, geom
    FROM current_vessel_state
    WHERE timestamp BETWEEN :t_min AND :t_max
      AND ST_DWithin(
          geom::geography,
          ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
          :radius_meters
      )
    UNION ALL
    SELECT mmsi, latitude, longitude, timestamp, geom
    FROM vessel_position_history
    WHERE timestamp BETWEEN :t_min AND :t_max
      AND ST_DWithin(
          geom::geography,
          ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
          :radius_meters
      )
) candidates
ORDER BY distance_meters ASC
LIMIT 1;
```
- **Correlation Outcome**:
  - Distance $\le 3000\text{ m}$ and $|T_{AIS} - T_{SAR}| \le 15\text{ min}$: `AIS-MATCHED` (records matched MMSI, distance in meters, and time delta in seconds).
  - No matching AIS candidate: `AIS-UNMATCHED` (explicit explanation: `"No AIS broadcast found within ±15m and 3.0km"`).
- **Strict Terminology**: Only `AIS-MATCHED` and `AIS-UNMATCHED`. Never "illegal vessel" or "criminal ship".

---

## 5. Single Unified Map Canvas

All three operational capabilities:
1. **Live Vessel Traffic** (AIS vessel markers with kinematic heading vectors and history trails)
2. **Weather-Aware Routing** (SeaRoute base route vs. weather-optimized route geometry, weather overlays)
3. **Dark-Vessel Intelligence** (Sentinel-1 radar scene footprint polygon, `AIS-MATCHED` cyan rings, `AIS-UNMATCHED` amber beacons)

operate simultaneously on the single unified MapLibre GL canvas without page reloads or conflicting coordinate systems. Control drawers float unobtrusively over the operational map.

