# Zero Out-of-Pocket Student Development Target & Dependency Audit

This document establishes the operational parameters, rate quotas, licensing terms, and fallback plans for all external dependencies. The core commitment of this project is a **zero out-of-pocket student development target**: no personal credit cards, no paid subscriptions, and no cloud vendor lock-in.

---

## 1. Provider-by-Provider Operational Audit

### 1.1 AISStream.io
- **Role**: Live real-time terrestrial and satellite AIS vessel position telemetry.
- **Free / Non-Commercial Status**: Free community access for personal, academic, and non-commercial development.
- **Current Usage Limits**: Rate-limited per IP/key; enforces spatial bounding-box filtering; bulk redistribution of raw feeds prohibited.
- **API Key Required?**: **YES** (Free registration).
- **Credit Card Required?**: **NO**.
- **SLA & Reliability Limitations**: No commercial uptime SLA; connections may experience brief resets during maintenance or server restarts.
- **Licensing & Attribution**: Attribution required in UI/documentation.
- **Fallback / Replacement Strategy**:
  1. Open Danish Maritime Authority (DMA) or Norwegian Coastal Administration (Kystverket) public AIS feeds.
  2. Local pre-recorded authentic NMEA 0183 / AIS JSON replay server (packaged in `scripts/ais_replay.py`).

### 1.2 Open-Meteo Marine API
- **Role**: Marine meteorological grids (waves, swell, ocean currents, surface winds).
- **Free / Non-Commercial Status**: Free for open-source and non-commercial projects.
- **Current Usage Limits**: Free tier allows up to **10,000 API requests per day** and **5,000 per hour**.
- **Demand / Cache-Driven Enforcement**: Weather is NOT continuously polled globally. Weather is fetched strictly on-demand for active route corridors and cached in Redis with a 6-hour TTL. A full route calculation typically consumes only 1 to 3 API requests, well within limits.
- **API Key Required?**: **NO**.
- **Credit Card Required?**: **NO**.
- **SLA & Reliability Limitations**: High availability community tier; subject to transient HTTP 429 throttling if burst limit is breached.
- **Licensing & Attribution**: Attribution mandatory: *"Marine weather forecasts provided by Open-Meteo.com"* under Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Fallback / Replacement Strategy**:
  1. Cached forecast grids in PostGIS.
  2. Direct ingestion of public NOAA WaveWatch III / GFS GRIB2 forecast files hosted on AWS Public Datasets (open egress, no credit card required).

### 1.3 Copernicus Data Space Ecosystem (CDSE) / Sentinel-1
- **Role**: Level-1 GRD Synthetic Aperture Radar (SAR) imagery for satellite dark-vessel detection.
- **Free / Non-Commercial Status**: Free public open access under the European Union Copernicus program.
- **Current Usage Limits**: Up to 4 concurrent downloads; generous daily download bandwidth allowance (tens of GB).
- **Event-Driven Enforcement**: Scene metadata queries (STAC) are separated from large raster asset downloads. The worker downloads raster bursts only when an authorized user explicitly selects a discovered scene footprint on the map.
- **API Key Required?**: **YES** (Free OAuth2 registration on CDSE portal).
- **Credit Card Required?**: **NO**.
- **SLA & Reliability Limitations**: Large GRD archives (~1.5 GB) take 30–90 seconds to download; STAC API can have occasional latency spikes.
- **Licensing & Attribution**: Open Access under EU Copernicus Sentinels Data Policy (CC-BY-SA 3.0 IGO).
- **Fallback / Replacement Strategy**: Pre-packaged open benchmark SAR scenes (e.g. English Channel, Dover Strait, Singapore) cached locally in `data/sample_sar/` for immediate verification without network dependence.

### 1.4 Vector Basemap Services (OpenFreeMap / Carto Positron)
- **Role**: Vector tiles for MapLibre GL JS (land, borders, labels).
- **Free / Non-Commercial Status**: Free vector tile services.
- **Current Usage Limits**: Public fair use under OpenStreetMap attribution.
- **API Key Required?**: **NO**.
- **Credit Card Required?**: **NO**.
- **SLA & Reliability Limitations**: Dependent on community server uptime.
- **Licensing & Attribution**: OpenStreetMap contributors (ODbL).
- **Fallback / Replacement Strategy**: Self-hosted local PMTiles vector archive served directly by local Nginx.

### 1.5 OpenSAR Insight SAR Vessel Dataset & Local YOLO26s Inference
- **Role**: Pre-training and held-out evaluation for the single SAR vessel detector.
- **Provider**: OpenSAR Insight team / Hugging Face.
- **Free / Open Access Status**: 100% free open access; public Hugging Face repository.
- **API Key Required?**: **NO**.
- **Credit Card Required?**: **NO**.
- **Local Inference Compute**: Runs entirely locally on CPU or Apple Silicon GPU (MPS) inside the Docker container with ~20.6 MB model weights.
- **Cost**: **$0.00**.

---

## 2. Compute Infrastructure & Student Cloud Portability

### 2.1 Canonical Local Docker Compose Environment (Mandatory Primary Target)
- **Cost**: **$0.00**.
- **Components**: PostgreSQL 16 + PostGIS, Redis 7, FastAPI Backend, Async Workers, React/Vite Frontend.
- **Hardware Profile**: Runs within 1.5–2.5 GB RAM on macOS, Linux, or Windows WSL2.
- **Zero Paid Dependencies**: Does not require any cloud account, domain registration, or external server.

### 2.2 Optional Student Cloud Credits (Non-Binding & Optional)
If external demonstration hosting is desired, the project can run on student credits, but **cloud credits are strictly optional and not an architectural dependency**:
- **DigitalOcean**: $200 free credit via GitHub Student Developer Pack (spins up a 4 GB RAM Basic Droplet for 6+ months).
- **Azure for Students**: $100 annual credit (no credit card required).
- **Oracle Cloud Free Tier**: 4 ARM Ampere cores + 24 GB RAM Always-Free VM.
- **Cloudflare Pages**: Free static frontend hosting with unlimited bandwidth.

### 2.3 ML Training Compute (Kaggle)
- **Cost**: **$0.00**.
- **Quota**: 30 hours per week of free NVIDIA T4 / P100 GPU compute.
- **Credit Card Required?**: **NO** (SMS verification only).
