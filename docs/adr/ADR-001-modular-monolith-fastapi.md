# ADR-001: Adoption of Modular Monolith Architecture with FastAPI

## Status
Accepted

## Context
Maritime Operations AI combines real-time streaming (AIS), heavy spatial calculations (weather routing), and satellite computer vision pipeline tasks (SAR dark-vessel detection). 

Starting with a multi-repo distributed microservice architecture introduces severe operational overhead: distributed tracing complexity, inter-service network latency, multi-container deployment orchestration costs, and schema synchronization friction. This overhead directly conflicts with our zero-budget student requirement and rapid development cadence.

Conversely, a classic monolithic design without strict internal boundaries quickly degenerates into tightly coupled "spaghetti code", making background processing, testing, and independent component evolution difficult.

## Decision
We adopt a **Modular Monolith** architecture implemented in Python 3.12+ with FastAPI:
1. The backend application codebase resides in a single unified repository (`backend/`).
2. Distinct domain modules (`vessel_traffic`, `weather_routing`, `sar_intelligence`) have isolated internal business logic, database schemas, and service interfaces.
3. Modules interact through clearly defined Python `typing.Protocol` interfaces or internal application service contracts, never by directly modifying another module's database tables or internal states.
4. Background workers execute as separate OS processes/entry points (`backend/app/workers/`) within the same repository container, sharing the domain logic and data access libraries without running inside the web request loop.

## Consequences
### Positive
- Single database migration pipeline (Alembic) and unified local development workflow (`docker compose up`).
- Zero inter-service network overhead for internal domain calls.
- High velocity and easy refactoring while maintaining strict boundaries.
- Seamless transition path: Any module can be extracted into an independent microservice in the future if team size or specific horizontal scaling requirements justify it.

### Negative / Trade-offs
- Developers must maintain discipline to prevent cross-importing private module symbols.
- All backend domains share the same Python runtime environment and dependencies.
