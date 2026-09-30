# Deployment & Operational Playbook

## 1. Overview & Strategy

The deployment model for **Maritime Operations AI** prioritizes:
1. **Local-First Verification**: 100% of services run via a standard `docker compose up` command on macOS, Linux, or Windows (WSL2) without cloud prerequisites.
2. **Zero Cloud Infrastructure Cost**: Deployable to free/student VPS credits (DigitalOcean, Azure, Oracle Cloud Free Tier) using standard Docker Compose without complex Kubernetes clusters.
3. **Reproducibility**: Multi-stage Docker builds ensure small container footprints and deterministic dependencies.

---

## 2. Containerized Service Topology

```mermaid
graph TD
    Client["User Browser"] -->|Port 80 / 443| Nginx["Nginx Reverse Proxy"]
    
    subgraph LocalStack["Docker Compose Local Network (maritime-net)"]
        Nginx -->|/api, /ws| Backend["FastAPI Backend Service<br/>(Uvicorn ASGI)"]
        Nginx -->|/| Frontend["React Frontend<br/>(Vite Static Assets)"]
        
        Backend --> Postgres[("PostgreSQL 16 + PostGIS 3.4<br/>(Port 5432)")]
        Backend --> Redis[("Redis 7 Alpine<br/>(Port 6379)")]
        
        AISWorker["AIS Ingestion Worker<br/>(Asyncio Daemon)"] --> Redis
        AISWorker --> Postgres
        
        WeatherWorker["Weather Sync Worker<br/>(Periodic Daemon)"] --> Redis
        SARWorker["SAR Pipeline Worker<br/>(On-Demand Daemon)"] --> Postgres
    end
    
    AISWorker <-->|TLS WebSocket| AISStream["AISStream.io"]
    WeatherWorker <-->|HTTPS| OpenMeteo["Open-Meteo Marine"]
    SARWorker <-->|HTTPS| CDSE["Copernicus CDSE"]
```

---

## 3. Service Configuration & Environment Specification

An example `.env.example` serves as the template for local and server environments:

```ini
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
APP_SECRET_KEY=change-in-production-use-a-strong-random-string

# Database (PostgreSQL + PostGIS)
POSTGRES_USER=maritime_user
POSTGRES_PASSWORD=maritime_secure_password
POSTGRES_DB=maritime_ops_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql+asyncpg://maritime_user:maritime_secure_password@postgres:5432/maritime_ops_db

# Cache & Message Broker (Redis)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://redis:6379/0

# External APIs
AISSTREAM_API_KEY=your_free_aisstream_api_key_here
COPERNICUS_CDSE_CLIENT_ID=optional_cdse_client_id
COPERNICUS_CDSE_CLIENT_SECRET=optional_cdse_client_secret

# Map basemaps
VITE_MAP_TILE_STYLE=https://basemaps.cartocdn.com/gl/positron-gl-style/style.json
```

---

## 4. Health Checks & Observability

### 4.1 Health Check & Diagnostic Endpoints
- `GET /api/health/live`: Liveness probe. Verifies that the Uvicorn process is responsive. Returns HTTP 200 `{"status": "alive"}`.
- `GET /api/health/ready`: Readiness probe. Verifies active database connectivity (`SELECT 1`) and Redis ping. Returns HTTP 200 `{"status": "ready", "database": "ok", "redis": "ok"}` or HTTP 503 if downstream dependencies fail.
- `GET /api/v1/vessels/pipeline/status`: Pipeline observability. Reports AISStream connection state, messages received, persisted count, malformed count, timestamp of last message, and total distinct vessels in PostGIS.

### 4.4 SAR & Dark-Vessel Diagnostics
- `GET /api/v1/sar/model/status`: Returns current YOLO26s SAR vessel detection model configuration, weight paths, and held-out test evaluation benchmarks.
- `GET /api/v1/sar/jobs/{job_id}`: Reports asynchronous SAR analysis job progress (`QUEUED`, `DOWNLOADING`, `PREPROCESSING`, `INFERENCE`, `CORRELATING`, `COMPLETE`), total tiles processed, and candidate detection counts.

### 4.5 Dedicated AIS Ingestion Daemon
The `ais-worker` service runs independently from FastAPI in `docker-compose.yml`:
```bash
# View AIS worker ingestion logs in real-time
docker compose logs -f ais-worker

# Restart AIS worker
docker compose restart ais-worker
```

### 4.6 Logging & Request Tracing
- Structured JSON logging using standard Python `logging`.
- Every incoming HTTP and WebSocket connection receives a unique UUID `X-Correlation-ID` to trace logs across API handlers, background stream consumers, and SAR jobs.

---

## 5. Database Migration Lifecycle

Database migrations are managed strictly through **Alembic**:
1. When models in `backend/app/infrastructure/database/models/` change, developers generate revisions:
   - Migration `0001_create_initial_schema.py` (vessel tables)
   - Migration `0002_create_routing_and_weather_tables.py` (routing & weather tables)
   - Migration `0003_create_sar_tables.py` (`sar_scenes`, `sar_detections`, `ais_sar_correlations`)
2. Migrations run automatically on container startup or via:
   ```bash
   docker compose exec backend alembic upgrade head
   ```

---

## 6. Zero Out-Of-Pocket Dark-Vessel Deployment

1. **Copernicus CDSE Account (Free)**:
   - Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/).
   - Set `COPERNICUS_CDSE_CLIENT_ID` and `COPERNICUS_CDSE_CLIENT_SECRET` in `.env` if automated raw raster downloads are desired.
   - Scene catalog search (`/api/v1/sar/search`) requires no credentials and queries the public STAC endpoint directly.
2. **Local Inference Execution**:
   - The YOLO26s model runs locally inside the Docker container on CPU or Apple Silicon MPS.
   - No external GPU instances, specialized inference cloud services, or paid APIs are used.
