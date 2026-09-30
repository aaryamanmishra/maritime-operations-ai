# ADR-001: Foundational Architecture Stack Selection

## Status
Accepted

## Context
Maritime Operations AI combines real-time streaming vessel telemetry, weather-aware multi-objective route optimization, and event-driven satellite radar (SAR) vessel detection. The project is constrained by a strict zero out-of-pocket student development target:
- Must run completely and predictably on local developer hardware at $0 cost.
- Must avoid distributed microservice complexity, cloud vendor lock-in, and unnecessary operational overhead.
- Must cleanly isolate high-frequency ingestion and asynchronous tasks from web request handling.
- Must provide spatial and kinematic query capabilities with sub-50ms viewport target benchmarks.

## Decision
We establish the core foundational stack:

1. **Modular Monolith with FastAPI (Python 3.12)**:
   - FastAPI provides high-throughput asynchronous request handling (`asyncio`), native OpenAPI schema generation, and robust data validation with Pydantic v2.
   - Organized as a modular monolith with clear domain boundaries (`vessel_traffic`, `ml_performance`, `weather_routing`, `sar_intelligence`) and asynchronous workers running as independent OS processes.
2. **PostgreSQL 16 with PostGIS 3.4**:
   - Primary durable spatial relational database.
   - PostGIS provides OGC-compliant spatial types (`Point`, `LineString`, `Polygon`), spatial indexing (`GIST`), geodetic distance calculations, and fast bounding-box queries for live vessel states and historical trajectories.
3. **Redis 7 (Alpine)**:
   - In-memory event broker and primary short-lived caching layer.
   - Used for Redis Streams (`stream:ais:raw`) to buffer high-frequency AIS sentences, Redis Pub/Sub for cross-process WebSocket fanout, and primary operational weather forecast caching with TTL.
4. **React 19 + TypeScript + Vite**:
   - Modern, high-performance web presentation layer.
   - Vite provides rapid HMR development and optimized static asset compilation.
   - TypeScript guarantees end-to-end type safety against backend API schemas.
5. **Docker Compose**:
   - Canonical local orchestration mechanism (`docker compose up --build`).
   - Ensures identical developer environments across macOS (Apple Silicon), Linux, and Windows WSL2 without manual dependency installation or cloud account setup.

## Consequences
### Positive
- Zero infrastructure spend required to develop, build, run, and test the entire platform.
- Simple, single-command startup with standard service discovery using Docker container hostnames (`postgres`, `redis`, `backend`, `frontend`).
- Strong domain modularity allowing future extraction of routing or ML services if scalability requires it.

### Negative / Trade-offs
- Docker Desktop must be running on the developer workstation.
- Development machines require at least 2–4 GB available RAM for the containerized stack.
