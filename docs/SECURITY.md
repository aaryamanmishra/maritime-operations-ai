# Security & Secrets Architecture

## 1. Secrets Management & Environment Isolation

### 1.1 Strict Server-Side Secrets Policy
- Under no circumstances will API tokens, database credentials, or secret keys be included in frontend bundles, client source files, or client-accessible environment files.
- The frontend Vite build is strictly configured to expose only variables prefixed with `VITE_PUBLIC_` (which are limited to public endpoint URLs).
- All sensitive credentials reside exclusively in the backend runtime environment, loaded via `pydantic-settings` from `.env` files that are strictly excluded from version control (`.gitignore`).

### 1.2 Handled Credentials
- `AISSTREAM_API_KEY`: Kept exclusively inside the backend AIS worker.
- `COPERNICUS_CDSE_CLIENT_ID` / `COPERNICUS_CDSE_CLIENT_SECRET`: Kept exclusively inside the SAR ingestion worker.
- `DATABASE_URL`: Asynchronous PostgreSQL connection string with password authentication.
- `REDIS_URL`: Redis authentication connection string.

---

## 2. Input Validation & Spatial Sanitization

### 2.1 Pydantic v2 Strong Typing
All incoming HTTP request payloads and WebSocket control frames are validated against strict Pydantic models before reaching domain logic:
- Latitude values must strictly satisfy $-90.0 \le lat \le 90.0$.
- Longitude values must strictly satisfy $-180.0 \le lon \le 180.0$.
- MMSI numbers must be validated as 9-digit integers adhering to ITU-R M.585-8 maritime identification formats.
- IMO numbers must satisfy the 7-digit check digit algorithm ($\sum_{i=1}^6 d_i \cdot (8-i) \pmod{10} = d_7$).
- Waypoint sequences are bounded to prevent Denial-of-Service (DoS) graph explosions (maximum 50 waypoints per routing request).

### 2.2 Spatial & SQL Injection Defense
- **Zero Raw Dynamic SQL**: All spatial database queries are constructed using SQLAlchemy 2.0 and GeoAlchemy2 or strictly parameterized SQL statements.
- Spatial geometries are instantiated exclusively through parameterized PostGIS functions (`ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)`), neutralizing spatial SQL injection vectors.

---

## 3. Network & Transport Security

### 3.1 HTTP & WebSocket Hardening
- **CORS (Cross-Origin Resource Sharing)**: Configured explicitly via FastAPI middleware. Disallows wildcard (`*`) origins in production, restricting access to designated frontend origins.
- **WebSocket Handshake Validation**: The WebSocket endpoint (`/ws/vessels`) validates origin headers, client connection rates, and max payload sizes (rejecting oversized messages $> 64\text{ KB}$).
- **Security Headers**: Injected by Nginx reverse proxy and FastAPI middleware:
  - `Content-Security-Policy (CSP)`: Restricts script, worker, and connect sources.
  - `Strict-Transport-Security (HSTS)`: Enforces TLS/HTTPS.
  - `X-Content-Type-Options: nosniff`.
  - `X-Frame-Options: DENY`.

### 3.2 Rate Limiting & DoS Mitigation
- The API applies in-memory or Redis-backed rate limiting (e.g. `slowapi`):
  - Standard REST read endpoints: 120 requests/minute per client IP.
  - Compute-heavy routing calculation endpoints: 10 requests/minute per client IP.
  - SAR scene search & detection endpoints: 5 requests/minute per client IP.

### 3.3 Information Leakage Prevention
- Global exception handlers catch unhandled runtime errors, log structured stack traces with request correlation IDs to the secure server log, and return clean, sanitized JSON error responses to the client (e.g., `{"error": "Internal Processing Error", "correlation_id": "abc-123"}`).
