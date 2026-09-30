# ADR-002: PostgreSQL and PostGIS as Durable Spatial Data Engine

## Status
Accepted

## Context
Maritime operations demand rigorous geospatial computations:
- Storing and querying high-velocity GPS/AIS vessel positions.
- Finding vessels within dynamic bounding boxes (spatial window queries).
- Spatial join operations between SAR satellite bounding footprints and AIS historical trajectories within temporal windows (±15 minutes).
- Storing nautical navigational route graphs, maritime choke points, exclusive economic zones (EEZs), and port coordinates.

Alternatives considered:
1. Pure Document Store (MongoDB): Lacks advanced spherical geodetic calculations and native indexing for complex maritime polygon intersections.
2. In-Memory Only (Redis GEO): Excellent for ephemeral live location lookup, but lacks durable history, spatial joins, trajectory simplification, and polygon topology operations.
3. PostgreSQL with PostGIS: Mature, open-source standard for spatial data, featuring R-tree GiST/SP-GiST indexing, geodetic geometry calculations (`geography` type), trajectory handling, and rich SQL analytical support.

## Decision
We select **PostgreSQL 16+ with the PostGIS 3.4+ extension** as our single durable spatial database engine.
- All vessel positions, historical tracks, voyage waypoints, bathymetry constraints, and SAR detection geometries are persisted with PostGIS spatial types (`geometry(Point, 4326)` and `geometry(LineString, 4326)`).
- GiST spatial indexes are enforced on all coordinate and track columns.
- Analytical queries leverage PostGIS functions (`ST_DWithin`, `ST_Intersects`, `ST_MakeLine`, `ST_SimplifyPreserveTopology`).

## Consequences
### Positive
- Production-grade spatial integrity and standards compliance (OGC / WGS84 EPSG:4326).
- Unrivaled spatial join performance for dark-vessel AIS vs. SAR correlation queries.
- Complete ecosystem compatibility with Python geospatial libraries (GeoAlchemy2, Shapely, PyGMT, GeoPandas).
- Available freely across all standard local and cloud development environments.

### Negative / Trade-offs
- PostGIS requires proper memory tuning (`shared_buffers`, `work_mem`) and spatial index maintenance (`VACUUM ANALYZE`) for high-volume AIS write workloads.
