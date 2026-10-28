# Dataset Provenance, Access Protocols, and Software Package Sources

This document details the exact provenance, licenses, official repository URLs, and citation protocols for all datasets, APIs, and open-source software libraries utilized in the **Maritime Operations AI** platform.

---

## 1. Primary Datasets and Remote Sensing Benchmarks

### 1.1 LS-SSDD-v1.0 (Large-Scale SAR Ship Detection Dataset)
- **Official Provider:** Key Laboratory of Digital Earth, Aerospace Information Research Institute, Chinese Academy of Sciences (CAS).
- **Primary Authors:** Tianwen Zhang, Xiaoling Zhang, Jun Li, et al.
- **Publication:** *Remote Sensing*, 12(19), 3167, 2020.
- **Repository Access:** Kaggle (`https://www.kaggle.com/datasets/petrarodriguez/ls-ssdd-v1-0`) and Baidu Netdisk.
- **Sensor Specification:** European Space Agency Copernicus Sentinel-1 C-band SAR ($5.405\,\text{GHz}$) in Interferometric Wide (IW) swath mode, dual-polarization (VV/VH).
- **Spatial Coverage:** 15 large-scale Sentinel-1 scenes covering $1,076\,\text{km} \times 1,280\,\text{km}$.
- **Dataset Composition:** 9,000 $800 \times 800$ pixel chips, 9,000 Pascal VOC XML annotation files, 15,487 ship targets.
- **License:** Academic & Non-Commercial Research Open Access.

### 1.2 OpenSAR Insight Benchmark Dataset
- **Official Provider:** Shanghai Jiao Tong University / OpenSAR Research Community.
- **Primary Authors:** Y. Li, S. Mao, and G. Li.
- **Publication:** *IEEE Transactions on Geoscience and Remote Sensing*, 59(8), 2021.
- **Repository Access:** Hugging Face Datasets (`https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset`).
- **Sensor Specification:** Sentinel-1 Single Look Complex (SLC) Level-1 slant-range chips.
- **Dataset Composition:** 3,573 chips ($640 \times 640$ pixels), 3,490 vessel bounding boxes.
- **License:** Open Access Research License.

### 1.3 Copernicus Data Space Ecosystem (CDSE)
- **Official Provider:** European Space Agency (ESA) & European Commission Copernicus Programme.
- **Service Portal:** `https://dataspace.copernicus.eu/`
- **STAC Catalog API:** `https://catalogue.dataspace.copernicus.eu/stac/`
- **OData Measurement API:** `https://zipper.dataspace.copernicus.eu/odata/v1/`
- **Authentication:** OAuth2 Keycloak Bearer Token (`https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`).
- **Products Utilized:** Sentinel-1 Level-1 Ground Range Detected High-Resolution (`S1C_IW_GRDH_1SDV_...`).
- **License:** Copernicus Open Access Policy (Free, full, and open data policy for all users).

### 1.4 AISStream Global Terrestrial & Satellite AIS Network
- **Official Provider:** AISStream.io Network.
- **Portal & API Documentation:** `https://aisstream.io/documentation`
- **Protocol:** Encrypted WebSocket connection (`wss://stream.aisstream.io/v0/stream`).
- **Message Types Ingested:** ITU-R M.1371 Messages 1, 2, 3 (Position Reports), Message 5 (Static Voyage Data).
- **License:** Developer API Terms of Service.

### 1.5 MARIS-Forecast NOAA Track A Vessel Performance Benchmark
- **Official Provider:** NOAA Office for Coastal Management / Zenodo Data Repository.
- **DOI:** [10.5281/zenodo.21224009](https://doi.org/10.5281/zenodo.21224009)
- **Features Included:** MMSI, SOG, COG, draft, wave height ($H_s$), wave period ($T_p$), wind speed, sea current.
- **License:** US Public Domain / Creative Commons CC0.

---

## 2. Core Software Packages and Open-Source Tooling

| Software Package | Version | Official Repository / Homepage | Open-Source License | Functional Role in System |
| :--- | :---: | :--- | :--- | :--- |
| **FastAPI** | `0.115.0` | `https://github.com/fastapi/fastapi` | MIT License | Asynchronous API gateway and WebSocket manager |
| **PostgreSQL** | `16.4` | `https://www.postgresql.org/` | PostgreSQL License | Relational database management system |
| **PostGIS** | `3.4.2` | `https://postgis.net/` | GNU GPL v2 | Spatial database extension (`GEOMETRY`, GIST index) |
| **Redis** | `7.2.5` | `https://redis.io/` | BSD 3-Clause | In-memory geospatial indexing (`GEOADD`) and Pub/Sub |
| **Ultralytics YOLO** | `8.3.0` | `https://github.com/ultralytics/ultralytics` | AGPL-3.0 License | YOLO11n, YOLO11s, YOLOv8s ship detection heads |
| **PyTorch** | `2.2.0` | `https://pytorch.org/` | Modified BSD | Deep-learning tensor framework and AMP engine |
| **GeoAlchemy2** | `0.15.0` | `https://github.com/geoalchemy/geoalchemy2` | MIT License | SQLAlchemy spatial integration for PostGIS types |
| **SQLAlchemy** | `2.0.35` | `https://www.sqlalchemy.org/` | MIT License | Async ORM and database connection pooling |
| **MapLibre GL JS** | `6.11.2` | `https://maplibre.org/` | BSD 3-Clause | WebGL-powered vector maritime map renderer |
| **React** | `19.0.0` | `https://react.dev/` | MIT License | Component-based frontend reactive user interface |
| **Vite** | `6.1.0` | `https://vitejs.dev/` | MIT License | Next-generation frontend build tool and dev server |
| **NumPy** | `2.4.6` | `https://numpy.org/` | BSD 3-Clause | Matrix percentile clipping and array manipulation |
| **Pillow (PIL)** | `10.4.0` | `https://python-pillow.org/` | HPND License | Image decoding, chip slicing, and array conversion |
| **Searoute** | `1.6.0` | `https://github.com/eurostat/searoute` | European Union Public Licence | Global maritime maritime graph routing engine |
| **Pytest** | `8.3.0` | `https://pytest.org/` | MIT License | Automated testing and verification test framework |
