# Maritime Operations AI — Teammate Onboarding Guide

**Repository:** `aaryamanmishra/maritime-operations-ai`  
**Verified Release:** Commit [`cbf4be70c6994b33802e0b325b3bb224fe2af967`](https://github.com/aaryamanmishra/maritime-operations-ai/commit/cbf4be70c6994b33802e0b325b3bb224fe2af967) (`main`)  
**Architecture:** Modular Monolith (FastAPI + React 19 + PostGIS + Redis 7 + Background Workers)

---

## 1. System Requirements & Environment

To run the complete platform locally:

| Requirement | Minimum Specification | Recommended |
| :--- | :--- | :--- |
| **Operating System** | macOS 13+ (Apple Silicon or Intel), Linux (Ubuntu 22.04+ / Debian 12+), Windows 10/11 with WSL2 | macOS (Apple Silicon) or Linux |
| **Container Engine** | Docker Engine 24.0+ & Docker Compose v2.20+ | Docker Desktop 4.25+ or Docker CE |
| **Memory (RAM)** | 8 GB free RAM | 16 GB RAM |
| **Storage** | 10 GB free disk space | 20 GB free disk space (for Docker images and cache) |
| **Local Runtimes** *(Optional, for host-side development)* | Python 3.12+ (64-bit), Node.js 20+ (LTS) & npm 10+ | Python 3.12.8, Node.js 22 LTS |

> [!NOTE]
> All core services (FastAPI backend, AIS worker, React frontend, PostgreSQL/PostGIS, Redis) run in Docker Compose. You do not need to install GDAL or PostGIS directly on your host operating system.

---

## 2. Repository Checkout

Clone the repository and verify your working tree matches the current release commit:

```bash
# Clone the repository
git clone git@github.com:aaryamanmishra/maritime-operations-ai.git
cd maritime-operations-ai

# Checkout and verify the verified release commit
git checkout cbf4be70c6994b33802e0b325b3bb224fe2af967

# Confirm clean status
git status
```

Expected output:
```text
HEAD detached at cbf4be7
nothing to commit, working tree clean
```
*(Or `On branch main` if tracking the remote main branch).*

---

## 3. Environment Configuration (`.env`)

Copy the template `.env.example` into `.env`:

```bash
cp .env.example .env
```

### Configuration Parameters Reference

| Variable Name | Default Value | Role & Usage | Required? |
| :--- | :--- | :--- | :---: |
| `ENVIRONMENT` | `development` | Sets logging and runtime mode in FastAPI. | No |
| `LOG_LEVEL` | `INFO` | Root logging verbosity (`DEBUG`, `INFO`, `WARNING`). | No |
| `APP_HOST` | `0.0.0.0` | Backend API bind address. | No |
| `APP_PORT` | `8000` | Backend API host port. | No |
| `CORS_ORIGINS` | `["http://localhost:5173", ...]` | Allowed origins for cross-origin browser requests. | No |
| `POSTGRES_USER` | `maritime_user` | PostgreSQL username used by backend and PostGIS container. | No |
| `POSTGRES_PASSWORD`| `maritime_secure_dev_password` | PostgreSQL password. | No |
| `POSTGRES_DB` | `maritime_ops` | Database name containing PostGIS extensions and tables. | No |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async SQLAlchemy connection string for FastAPI and worker. | No |
| `REDIS_URL` | `redis://redis:6379/0` | Connection string for Redis Streams and weather cache. | No |
| `VITE_API_BASE_URL`| `http://localhost:8000` | Target backend URL compiled into frontend bundles. | No |
| `AISSTREAM_API_KEY`| *(empty)* | Free API key from [aisstream.io](https://aisstream.io) for live global AIS ingestion. | **Optional** |
| `AIS_BBOX_MIN_LAT` | `49.5` | Southern latitude for AIS stream geofence (Dover Strait default). | No |
| `AIS_BBOX_MIN_LON` | `-2.0` | Western longitude for AIS stream geofence. | No |
| `AIS_BBOX_MAX_LAT` | `51.5` | Northern latitude for AIS stream geofence. | No |
| `AIS_BBOX_MAX_LON` | `2.5` | Eastern longitude for AIS stream geofence. | No |
| `CDSE_USERNAME` | *(empty)* | Copernicus Data Space username (free registration). | **Optional** |
| `CDSE_PASSWORD` | *(empty)* | Copernicus Data Space password. | **Optional** |
| `CDSE_CLIENT_ID` | *(empty)* | Alternative OAuth2 Keycloak Client ID. | **Optional** |
| `CDSE_CLIENT_SECRET`| *(empty)* | Alternative OAuth2 Keycloak Client Secret. | **Optional** |
| `SAR_MODEL_WEIGHTS_PATH` | `backend/ml/weights/yolo26s_sar_vessel.pt` | Path to trained YOLO26s GRD weights. | Pre-tracked |
| `SAR_SLC_MODEL_WEIGHTS_PATH` | `backend/ml/weights/yolo26s_slc_vessel.pt` | Path to trained YOLO26s SLC weights. | Pre-tracked |

> [!IMPORTANT]
> **Zero Secrets Policy**: Never commit `.env`, session JSON files (`.sar_s3_session.json`), or API tokens to git. `.env` and `.sar_s3_session.json` are strictly excluded by `.gitignore`.

---

## 4. Pre-Tracked Model Assets & Reference Data

All necessary machine learning models, weights, metadata, and geographic vectors are **already committed and tracked** inside the repository. Teammates do not need to download weights from external cloud buckets or retrain models:

| Asset Path | Size | Description |
| :--- | :---: | :--- |
| [`backend/ml/weights/yolo26s_sar_vessel.pt`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/ml/weights/yolo26s_sar_vessel.pt) | ~21.6 MB | Ultralytics YOLO26s detector trained on OpenSAR Insight Sentinel-1 GRD. |
| [`backend/ml/weights/yolo26s_metadata.json`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/ml/weights/yolo26s_metadata.json) | ~1.7 KB | GRD model hyperparameters, class definitions, and held-out test evaluation metrics. |
| [`backend/ml/weights/yolo26s_slc_vessel.pt`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/ml/weights/yolo26s_slc_vessel.pt) | ~20.3 MB | Ultralytics YOLO26s detector trained on OpenSAR Insight Sentinel-1 SLC complex magnitude. |
| [`backend/ml/weights/yolo26s_slc_metadata.json`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/ml/weights/yolo26s_slc_metadata.json) | ~830 B | SLC model training parameters and held-out test split evaluation metrics. |
| [`backend/ml/weights/tft_vessel_performance.pt`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/ml/weights/tft_vessel_performance.pt) | ~641 KB | Temporal Fusion Transformer surrogate model predicting speed loss under weather stress. |
| [`backend/ml/weights/tft_metadata.json`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/ml/weights/tft_metadata.json) | ~2.9 KB | TFT feature scalers, input encodings, and performance validation metrics. |
| [`backend/app/domain/sar_intelligence/data/ne_10m_land.json`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/app/domain/sar_intelligence/data/ne_10m_land.json) | ~7.36 MB | Natural Earth 1:10m physical coastline vectors for terrestrial radar clutter rejection. |
| [`backend/app/domain/sar_intelligence/data/ne_110m_land.json`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/app/domain/sar_intelligence/data/ne_110m_land.json) | ~237 KB | Low-memory continental fallback vector layer. |
| [`backend/app/domain/sar_intelligence/data/burst_258592_metadata.json`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/app/domain/sar_intelligence/data/burst_258592_metadata.json) | ~14 KB | Offline SLC burst GCP grid and incidence angle metadata fixture for testing. |
| [`backend/app/domain/sar_intelligence/data/slc_test_chip.jpg`](file:///Users/aaryamanmishra/Documents/maritime-operations-ai/backend/app/domain/sar_intelligence/data/slc_test_chip.jpg) | ~24 KB | Synthetic radar target chip for deterministic offline pipeline regression tests. |

Inside Docker Compose, the repository's `./backend` directory is mounted to `/app`, ensuring all weights and datasets are accessible inside containers without manual copying.

---

## 5. Docker Compose Operations

### Starting the Platform

Build and launch all 5 containers in detached mode:

```bash
docker compose up -d --build
```

### Checking Container Health

Verify that all containers report `Up` and `healthy`:

```bash
docker compose ps
```

Expected output:
```text
NAME                  SERVICE      STATUS                    PORTS
maritime_ais_worker   ais-worker   Up                        8000/tcp
maritime_backend      backend      Up (healthy)              0.0.0.0:8000->8000/tcp
maritime_frontend     frontend     Up                        0.0.0.0:5173->80/tcp
maritime_postgres     postgres     Up (healthy)              0.0.0.0:5432->5432/tcp
maritime_redis        redis        Up (healthy)              0.0.0.0:6379->6379/tcp
```

### Viewing Logs

```bash
# Tail all container logs
docker compose logs -f

# Tail specific services
docker compose logs -f backend
docker compose logs -f ais-worker
docker compose logs -f frontend
```

### Stopping the Platform

```bash
# Stop containers while preserving database and Redis volumes
docker compose down

# Stop containers and wipe database volumes (clean reset)
docker compose down -v
```

---

## 6. Access Endpoints & Health Verification

Once containers are healthy, test the endpoints:

| Service | Target URL | Expected Response |
| :--- | :--- | :--- |
| **Web UI** | `http://localhost:5173` | React 19 interactive MapLibre map shell (`HTTP 200`). |
| **FastAPI Root** | `http://localhost:8000/` | `{"project": "Maritime Operations AI", "version": "0.1.0", "status": "operational"}` |
| **Readiness Probe** | `http://localhost:8000/api/health/ready` | `{"status": "ready", "database": "ok", "redis": "ok"}` |
| **Liveness Probe** | `http://localhost:8000/api/health/live` | `{"status": "alive"}` with `X-Correlation-ID` header. |
| **Swagger API Docs** | `http://localhost:8000/docs` | Interactive OpenAPI documentation. |

### Database Migrations Verification

Database schema migrations are applied automatically on backend startup via `alembic upgrade head`. Verify the current revision:

```bash
docker compose exec backend alembic current
```
Expected output:
```text
0003_create_sar_tables (head)
```

Verify PostGIS extension in PostgreSQL:
```bash
docker compose exec postgres psql -U maritime_user -d maritime_ops -c "SELECT postgis_full_version();"
```

---

## 7. End-to-End Workflow Verification

### A. Live AIS Traffic

Check the status of the AIS ingestion worker:

```bash
curl -s http://localhost:8000/api/v1/vessels/pipeline/status | python3 -m json.tool
```

Query vessels currently inside the Dover Strait bounding box:

```bash
curl -s "http://localhost:8000/api/v1/vessels?min_lat=49.5&min_lon=-2.0&max_lat=51.5&max_lon=2.5" | python3 -c "
import sys, json
vessels = json.load(sys.stdin)
print(f'Vessels in viewport: {len(vessels)}')
if vessels:
    print('Sample vessel:', json.dumps(vessels[0], indent=2))
"
```

> [!NOTE]
> If you have not provided an `AISSTREAM_API_KEY`, the worker logs a warning and will not ingest new live packets. Historical vessels seeded into PostGIS will still be returned by queries.

### B. Weather-Aware Routing Optimization

Execute a route optimization request between Brest, France and Dover, UK:

```bash
curl -s -X POST http://localhost:8000/api/v1/routing/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "origin": {"latitude": 49.0, "longitude": -3.0},
    "destination": {"latitude": 51.0, "longitude": 1.5},
    "departure_time": "2026-10-21T12:00:00Z",
    "vessel": {
      "vessel_type": "cargo",
      "length_m": 190.0,
      "beam_m": 28.0,
      "draft_m": 10.5,
      "design_speed_kn": 14.5
    },
    "fuel": {
      "fuel_price_per_tonne": 620.0,
      "fuel_type": "VLSFO"
    },
    "optimization_priority": "balanced"
  }' | python3 -c "
import sys, json
res = json.load(sys.stdin)
base = res['base_route']['summary']
print(f\"Route calculated successfully!\")
print(f\"Distance: {base['total_distance_nm']:.1f} nm\")
print(f\"Estimated Duration: {base['estimated_duration_hours']:.1f} h\")
print(f\"Fuel: {base['total_estimated_fuel_tonnes']:.2f} tonnes VLSFO\")
print(f\"Estimated Cost: \${base['total_estimated_cost_usd']:,.2f}\")
print(f\"Marine Weather: Wave Height {base['max_wave_height_m']:.2f}m, Exposure: {base['weather_exposure_level']}\")
"
```

### C. Satellite Radar (SAR) Dark-Vessel Detection

Check operational model status for both GRD and SLC checkpoints:

```bash
# Verify default GRD model
curl -s http://localhost:8000/api/v1/sar/model/status | python3 -m json.tool

# Verify opt-in SLC model
curl -s "http://localhost:8000/api/v1/sar/model/status?modality=SLC" | python3 -m json.tool
```

Submit a demo GRD analysis job (runs locally using pre-packaged weights and sample raster):

```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/sar/jobs \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "S1A_IW_GRDH_DEMO_SCENE", "modality": "GRD"}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

echo "Created job: $JOB_ID"
sleep 3
curl -s "http://localhost:8000/api/v1/sar/jobs/$JOB_ID" | python3 -m json.tool
```

Verify that real SLC acquisitions fail safely without CDSE credentials:

```bash
SLC_JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/sar/jobs \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "S1A_IW_SLC__1SDV_20240810T000834", "modality": "SLC"}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

sleep 3
curl -s "http://localhost:8000/api/v1/sar/jobs/$SLC_JOB_ID" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Status:', data['status'])
print('Message:', data['message'])
print('Detections:', len(data['detections']))
"
```
Expected output:
```text
Status: FAILED
Message: SAR analysis failed: ...
Detections: 0
```

---

## 8. External Connectivity & Credential Boundaries

| Platform Feature | External Connectivity Required? | Credentials Required? | Failure / Offline Behavior |
| :--- | :---: | :---: | :--- |
| **Web UI & MapLibre Map** | Yes (CDN for basemap tiles) | None | Uses Carto Dark Matter public tiles; application HUD loads offline. |
| **Live AIS Ingestion** | Yes (`wss://stream.aisstream.io`) | `AISSTREAM_API_KEY` (Free) | Logs clean warning; uses existing PostGIS records. |
| **Marine Weather Routing** | Yes (`marine-api.open-meteo.com`) | None (Public API) | Queries hourly wave/wind data; cached in Redis with 6h TTL. |
| **SAR Catalog Discovery** | Yes (`stac.dataspace.copernicus.eu`)| None (Public STAC) | Returns metadata for Sentinel-1 scenes without credentials. |
| **SAR GRD Local Analysis** | **No** (Fully Offline) | None | Operates on pre-tracked YOLO26s GRD weights and local sample images. |
| **SAR SLC Live Burst Ingestion** | Yes (`eodata.dataspace.copernicus.eu`) | `CDSE_USERNAME` + `CDSE_PASSWORD` | **Fails safely to `FAILED`** status; never produces synthetic detections. |

---

## 9. Running Tests, Linters, and Builds

### Backend Testing (Inside Docker)

```bash
# Run the complete backend test suite (68 tests)
docker compose exec backend pytest tests/

# Run specific domain test suites
docker compose exec backend pytest tests/test_sar_slc_pipeline.py
docker compose exec backend pytest tests/test_sar_intelligence.py tests/test_sar_geographic_context.py
docker compose exec backend pytest tests/test_weather_routing.py

# Run Ruff linting
docker compose exec backend ruff check app tests
```

### Frontend Testing & Build (Host Machine)

Requires Node.js 20+:

```bash
cd frontend

# Install dependencies
npm install

# Run Vitest suite (9 tests)
npm test

# Run TypeScript type check and production Vite build
npm run build
```

---

## 10. Troubleshooting & Common Pitfalls

### 1. Port Collisions (Ports 5432, 6379, 8000, 5173)
- **Symptom**: `Bind for 0.0.0.0:5432 failed: port is already allocated`.
- **Cause**: A local instance of PostgreSQL, Redis, or Vite is running on your host machine.
- **Fix**: Stop the local host service (`brew services stop postgresql`, `brew services stop redis`), or change the mapped host port in `docker-compose.yml`.

### 2. Database Connection Role Errors
- **Symptom**: `psql: error: FATAL: role "postgres" does not exist`.
- **Cause**: The project uses a dedicated role `maritime_user` rather than the default superuser `postgres`.
- **Fix**: Always specify `-U maritime_user -d maritime_ops` when executing psql commands.

### 3. AIS Worker Connection Warning
- **Symptom**: `ais_worker.py: AISSTREAM_API_KEY is not configured`.
- **Cause**: No key provided in `.env`.
- **Fix**: This is harmless for offline development. If live streaming is needed, register for a free key at [aisstream.io](https://aisstream.io) and set `AISSTREAM_API_KEY=your_key` in `.env`, then run `docker compose restart ais-worker`.

### 4. Git Diff Line Ending / Whitespace Issues
- **Symptom**: `git diff --check` reports trailing whitespace errors.
- **Fix**: Run `git diff --check` before committing. Ensure code editors do not add trailing spaces.

---

## 11. Known System Limitations & Operational Notices

1. **SLC Detection Accuracy Benchmark Notice**:
   - The YOLO26s SLC model is trained on OpenSAR Insight complex amplitude slices and achieves `0.5255` precision and `0.4164` recall on the held-out OpenSAR test split.
   - **Operational Notice**: Full-scene SLC detection accuracy has **not yet been independently benchmarked against external labeled ground truth** (such as concurrent AIS broadcast logs or optical satellite cross-verification). Real-world IW SLC acquisitions may exhibit azimuth ambiguities, range sidelobes, or coastal speckle. Target detections are operational indicators, not certified navigational ground truth.
2. **Snapshot Radar vs. Live Radar**:
   - Satellite radar observations are discrete, timestamped snapshots taken at the moment of satellite overpass. They must never be described as "real-time radar" or "live SAR".
3. **Strict Attribution Terminology**:
   - Non-reporting radar contacts must be classified strictly as `AIS-UNMATCHED` and correlated contacts as `AIS-MATCHED`.
   - Never apply prejudicial labels such as "illegal vessel" or "dark fleet". An `AIS-UNMATCHED` status simply means no AIS broadcast matched the radar observation within $\pm 15$ minutes and $3.0\text{ km}$ under the configured matching rules.
4. **Voyage Optimization Advisory**:
   - Weather-aware route trajectories, speed loss predictions, and fuel savings are computational estimates derived from naval architecture physics (Holtrop-Mennen, Kwon) and a Temporal Fusion Transformer surrogate model. They do not constitute certified voyage plans or safety-of-life-at-sea guarantees.
