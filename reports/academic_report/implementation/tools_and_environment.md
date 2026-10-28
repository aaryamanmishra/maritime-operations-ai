# Tools, Software Environment, and Reproducibility Specification

This document details the complete hardware, operating system, containerized software environment, dependency inventory, and execution commands required to reproduce the **Maritime Operations AI** platform and its experimental findings.

---

## 1. Environment Topology & Separation of Runtimes

The system architecture distinguishes between three operational runtimes:

```
+-------------------------------------------------------------------------------------------------+
|                               SYSTEM ENVIRONMENT TOPOLOGY                                       |
+--------------------------+------------------------------+---------------------------------------+
| 1. Local Workstation     | Windows 11 Pro / WSL2 Ubuntu | Host machine running Docker Desktop   |
| 2. Containerized Stack   | 5 Isolated Docker Containers | Backend, Worker, DB, Redis, Frontend  |
| 3. Cloud Training Fleet  | Kaggle NVIDIA Tesla T4 GPU   | Multi-epoch YOLO model training & AMP |
+--------------------------+------------------------------+---------------------------------------+
```

> [!IMPORTANT]
> Configurations and dependencies differ across these environments. Cloud GPU runtimes utilize CUDA 12.2/13.0 for accelerated mixed-precision PyTorch training, while the local Docker container runtime executes on containerized Linux Alpine/Debian targeting CPU inference and microservice orchestration.

---

## 2. Environment Specifications

### 2.1 Local Host Development Environment
- **Operating System:** Microsoft Windows 11 Pro 64-bit (Build 22631.4317)
- **Subsystem:** Windows Subsystem for Linux (WSL2, Ubuntu 22.04 LTS kernel 5.15.153.1)
- **Host CPU:** Intel Core i7-1185G7 @ 3.00 GHz (4 Cores, 8 Logical Processors)
- **Host RAM:** 32.0 GB DDR4-3200 MHz
- **Host Python:** Python 3.11.9 (Local tooling)
- **Host Node.js:** Node.js v20.18.0 / npm 10.8.2
- **Container Engine:** Docker Desktop 4.35.0 (Docker Engine 27.2.0, Compose v2.29.7)

### 2.2 Containerized Application Runtime (Docker)
- **Base Linux Distribution:** Python 3.12-slim-bookworm (Debian 12)
- **Backend Runtime:** Python 3.12.15
- **Web API Framework:** FastAPI 0.115.0
- **ASGI Server:** Uvicorn 0.31.0
- **Spatial Database:** PostgreSQL 16.4 with PostGIS 3.4.2 (`postgis/postgis:16-3.4-alpine`)
- **In-Memory Cache:** Redis 7.2.5 (`redis:7-alpine`)
- **Frontend Container:** Node.js 20 Alpine / Nginx 1.27 Alpine

### 2.3 Cloud Model Training Environment (Kaggle)
- **Accelerator:** NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM, 2560 CUDA cores)
- **Host CPU:** Intel Xeon @ 2.20 GHz (2 vCPUs, 13.0 GB RAM)
- **CUDA Version:** CUDA 12.2 / Driver 535.104.05
- **PyTorch Stack:** PyTorch 2.2.0+cu121, TorchVision 0.17.0+cu121
- **ML Engine:** Ultralytics YOLO 8.3.0

---

## 3. Complete Package Dependency Inventory

### 3.1 Backend Python Dependencies (`backend/pyproject.toml`)
```toml
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.31.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "sqlalchemy[asyncio]>=2.0.35",
    "greenlet>=3.0.3",
    "asyncpg>=0.29.0",
    "geoalchemy2>=0.15.0",
    "alembic>=1.13.0",
    "redis>=5.1.0",
    "httpx>=0.27.0",
    "websockets>=13.0",
    "searoute>=1.6.0",
    "scipy>=1.12.0",
    "torch>=2.2.0",
    "opencv-python-headless>=4.10.0",
    "ultralytics>=8.3.0",
    "numpy>=1.26.0",
    "pillow>=10.4.0",
    "pyyaml>=6.0.2"
]
dev_dependencies = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0"
]
```

### 3.2 Frontend Dependencies (`frontend/package.json`)
```json
{
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "maplibre-gl": "^6.11.2"
  },
  "devDependencies": {
    "typescript": "~5.7.2",
    "vite": "^6.1.0",
    "vitest": "^3.0.5",
    "tailwindcss": "^3.4.19",
    "postcss": "^8.5.28",
    "autoprefixer": "^10.6.1",
    "@types/geojson": "^7946.0.16",
    "@testing-library/react": "^16.2.0",
    "@testing-library/jest-dom": "^6.6.3",
    "jsdom": "^26.0.0"
  }
}
```

---

## 4. Environment Variables Configuration Inventory

All sensitive configuration parameters are managed via environment variables. In adherence to security protocols, secret values are sanitized below:

| Variable Name | Required | Default / Example | Purpose in Maritime Operations AI |
| :--- | :---: | :--- | :--- |
| `ENVIRONMENT` | Yes | `development` | Runtime mode (`development`, `staging`, `production`) |
| `LOG_LEVEL` | Yes | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DATABASE_URL` | Yes | `postgresql+asyncpg://[USER]:[PASS]@postgres:5432/[DB]` | Asynchronous PostgreSQL connection string |
| `REDIS_URL` | Yes | `redis://redis:6379/0` | In-memory cache and pub/sub broker URI |
| `CORS_ORIGINS` | Yes | `["http://localhost:5173", "http://localhost:3000"]` | Permitted Cross-Origin Resource Sharing origins |
| `AISSTREAM_API_KEY` | Yes | `[SANITIZED_API_KEY]` | Authentication key for AISStream.io live WebSocket |
| `CDSE_USERNAME` | Optional | `[SANITIZED_EMAIL]` | Copernicus Data Space Ecosystem username |
| `CDSE_PASSWORD` | Optional | `[SANITIZED_PASSWORD]` | Copernicus Data Space Ecosystem password |
| `CDSE_CLIENT_ID` | Optional | `[SANITIZED_CLIENT_ID]` | CDSE Keycloak OAuth2 Client ID |
| `CDSE_CLIENT_SECRET` | Optional | `[SANITIZED_SECRET]` | CDSE Keycloak OAuth2 Client Secret |
| `AIS_BBOX_MIN_LAT` | Yes | `49.5` | Southern latitude bound for AIS ingestion filter |
| `AIS_BBOX_MIN_LON` | Yes | `-2.0` | Western longitude bound for AIS ingestion filter |
| `AIS_BBOX_MAX_LAT` | Yes | `51.5` | Northern latitude bound for AIS ingestion filter |
| `AIS_BBOX_MAX_LON` | Yes | `2.5` | Eastern longitude bound for AIS ingestion filter |
| `SAR_CORRELATION_TIME_WINDOW_MINUTES` | No | `15` | Temporal window for AIS/SAR correlation ($\pm 15$ min) |
| `SAR_CORRELATION_DISTANCE_KM` | No | `3.0` | Base spatial matching threshold in kilometers |

---

## 5. System Setup, Training, and Verification Commands

### 5.1 Microservice Stack Startup (Docker Compose)
```bash
# 1. Clone repository and navigate to root directory
cd maritime-operations-ai

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env to supply CDSE and AISStream credentials

# 3. Build container images and launch microservices
docker compose up -d --build

# 4. Verify running container health
docker ps
# maritime_backend (8000), maritime_frontend (5173), maritime_postgres (5432), maritime_redis (6379), maritime_ais_worker
```

### 5.2 Database Migrations
```bash
# Execute Alembic asynchronous schema migrations inside the backend container
docker exec -it maritime_backend alembic upgrade head
```

### 5.3 Model Training Commands (Kaggle Cloud GPU)
```bash
# Execute LS-SSDD-v1.0 radar-adapted training
python backend/train_lsssd_detector.py \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --base-model yolo11s.pt \
    --full-source /kaggle/working/yolo_lsssd_full \
    --curated-dir /kaggle/working/yolo_lsssd_curated \
    --run-dir /kaggle/working/runs
```

### 5.4 Automated Test Suite Execution
```bash
# Execute complete backend pytest regression suite inside container
docker exec -it maritime_backend pytest -v

# Run with execution timing profile
docker exec -it maritime_backend pytest --durations=10

# Execute frontend Vitest test suite
cd frontend && npm run test
```
