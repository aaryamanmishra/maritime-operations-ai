# ADR-007: Zero-Cost and Free-First Student Infrastructure Strategy

## Status
Accepted

## Context
The project developer has essentially zero budget. Paid cloud databases, managed Redis clusters, enterprise satellite imagery feeds (Planet, Spire, Maxar), and proprietary marine weather APIs (StormGeo, WeatherNews) would render this project unsustainable and unbuildable for a student.

Furthermore, cloud free tiers often come with traps: hidden egress fees, mandatory credit card input with unexpected billing spikes, automatic sleep/cold starts that break persistent WebSockets, and abrupt quota throttling.

## Decision
We enforce a **Strict Free-First Architecture**:
1. **Local-First Containerized Stack**: The primary development and execution environment is 100% locally self-hosted via **Docker Compose**:
   - Web API (FastAPI)
   - Background Workers (Python asyncio)
   - Spatial Database (PostgreSQL 16 + PostGIS 3.4)
   - In-Memory Broker (Redis 7 Alpine)
   - Client App (Vite Dev Server / Static Nginx)
   This ensures complete feature functionality with zero cloud spend and zero network egress cost.
2. **Free External APIs Only**:
   - AIS: **AISStream.io** (Free community WebSocket tier; requires free API key; strictly throttled client-side; no credit card required).
   - Weather: **Open-Meteo Marine API** (Free for non-commercial/open source use; no API key required; rate-limited to 10,000 daily requests; no credit card required).
   - Satellite SAR: **Copernicus Data Space Ecosystem (CDSE)** (Free public access to Sentinel-1; generous daily download allowance; no credit card required).
   - Basemap Tiles: **OpenFreeMap** / **CartoCDN Open** (Free vector tile servers; no credit card required; fallbacks to local PMTiles vector caches).
3. **Free ML Training Infrastructure**:
   - Kaggle Kernels (30 hours/week free NVIDIA T4/P100 GPUs) for SAR model training.
   - GitHub Releases or Hugging Face Hub (free model weight hosting).
4. **Cloud Deployment Targets (Optional Zero-Cost Hosting)**:
   - Frontend: Cloudflare Pages / Vercel (Free tier, unlimited bandwidth, global CDN).
   - Backend API & DB: Self-hosted VPS via GitHub Student Developer Pack credits (DigitalOcean $200 / Azure $100 student credits), or local Docker. For zero-credit scenarios, local Docker Compose is the verified canonical reference.

## Consequences
### Positive
- The application can be completely spun up and verified by any student on any standard computer (macOS, Linux, Windows WSL) without spending a cent.
- No risk of accidental cloud billing or surprise credit card charges.
- Architecture remains modular: any component can be swapped for an enterprise provider (e.g. Spire AIS or AWS RDS) by simply altering environment variables.

### Negative / Trade-offs
- Free AISStream WebSocket may occasionally drop or restart during high-load periods, requiring robust client-side reconnection logic.
- Satellite scene downloads from CDSE are subject to free tier queue concurrency limits.
