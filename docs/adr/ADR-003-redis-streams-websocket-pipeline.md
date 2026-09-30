# ADR-003: Redis Streams and WebSockets for Real-Time Event Pipeline

## Status
Accepted

## Context
AIS telemetry arrives at irregular, high-frequency intervals (hundreds to thousands of messages per second in active waterways). Directly pushing each received AIS message into the relational database and concurrently broadcasting to active browser clients would overwhelm database connection pools and degrade map performance.

Furthermore, we anticipate scaling the backend to multiple FastAPI worker processes or nodes in the future. In-memory Python pub/sub solutions cannot cross process or container boundaries.

## Decision
We implement a two-tiered real-time architecture utilizing **Redis 7+ Streams and Pub/Sub**:
1. **Ingestion Buffer**: The dedicated `ais_ingestion_worker` reads raw AIS frames, converts them into canonical JSON events, and appends them to a Redis Stream (`stream:ais:raw`) using `XADD` with an approximate cap (`MAXLEN ~ 100000`).
2. **Persistence Micro-Batching**: A consumer group (`group:ais:db_persist`) reads from the stream in batches of up to 500 items or 1-second intervals, executing high-throughput multi-row `INSERT ... ON CONFLICT DO UPDATE` statements into PostgreSQL.
3. **Real-time Client Broadcast**: A separate Redis Pub/Sub channel (`channel:ais:broadcast`) distributes real-time kinematic updates to all connected FastAPI instances.
4. **Spatial Viewport Throttling**: Connected frontend clients open a single persistent WebSocket connection (`/ws/vessels`) and transmit their bounding box. The server tracks client viewport subscriptions in memory and throttles updates (e.g., maximum 1 update per vessel every 3 seconds per client), preventing browser memory exhaustion.

## Consequences
### Positive
- Total decoupling between external socket ingestion, persistence workloads, and client push sockets.
- Backpressure resistance: If the database is under heavy maintenance or load, the Redis Stream safely buffers incoming AIS bursts.
- Multi-instance ready: New API pods can join and stream updates immediately without coordinating with the ingestion worker.

### Negative / Trade-offs
- Introduces Redis as a mandatory operational dependency.
- Requires managing Redis memory limits and stream compaction policies.
