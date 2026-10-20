# SAR Intelligence Geographic Reference Data

This directory contains land boundary reference datasets used by `GeographicContextValidator` in `backend/app/domain/sar_intelligence/geographic_context.py` to validate candidate detections, suppress continental terrestrial radar clutter, and disambiguate coastal and port vessels.

## Datasets

### 1. `ne_10m_land.json` (Primary)
- **Source**: [Natural Earth](https://www.naturalearthdata.com/) - 1:10m Physical Vectors (`ne_10m_land`)
- **Version**: 5.1.1
- **License**: Public Domain (CC0 / No copyright restrictions)
- **Format**: GeoJSON `FeatureCollection` with 4,064 Polygon and MultiPolygon geometries
- **Size**: ~7.36 MB uncompressed
- **Precision**: 4 decimal places (approx. 11 m resolution at equator), matching high-resolution SAR spatial bounds without overhead.
- **Purpose**: High-fidelity coastline delineation resolving intricate coastal geometries (e.g., Lagos / Niger Delta, Port of Esbjerg / Wadden Sea, Cameroon estuary, Douala, Singapore, Rotterdam). Resolves coastal vessels previously misclassified under coarse 1:110m boundaries.

### 2. `ne_110m_land.json` (Fallback)
- **Source**: [Natural Earth](https://www.naturalearthdata.com/) - 1:110m Physical Vectors (`ne_110m_land`)
- **Version**: 5.1.1
- **License**: Public Domain (CC0 / No copyright restrictions)
- **Format**: GeoJSON `FeatureCollection` with 127 simplified continental polygons
- **Size**: ~237 KB
- **Purpose**: Fast, low-memory continental fallback if 1:10m data is missing.

## Attribution
Natural Earth vector map data is in the public domain. Commercial and non-commercial use is permitted without royalties or licensing fees.
