# Georeferencing, Coordinate Normalization, and AIS Correlation Specification

This document provides the mathematical derivations, coordinate reference system conventions, algorithmic specifications, and empirical validation results for the geospatial georeferencing and kinematic AIS correlation subsystems of the **Maritime Operations AI** platform.

---

## 1. Operational Sequencing: SAR First, AIS Correlation Second

A critical conceptual foundation of Maritime Operations AI is its **sensor-driven operational sequence**:
1. **SAR Spaceborne Detection (Primary Evidence):** The satellite radar pass operates as the unalterable, non-cooperative physical observation. Sentinel-1 illuminates the ocean surface and detects metallic/composite ship hulls regardless of whether the vessel wishes to be detected.
2. **AIS Telemetry Correlation (Secondary Context):** Once a potential surface vessel is localized and georeferenced, historical and streaming AIS records are queried to determine whether a cooperative broadcast accounts for the radar contact.
3. **Dark-Vessel Identification:** If a confirmed radar contact exhibits zero matching AIS broadcasts within physically plausible kinematic boundaries, the contact is flagged as an `AIS-UNMATCHED` potential dark vessel.

---

## 2. Coordinate Systems, Conventions, and Axis-Ordering Invariants

To eliminate the common "axis swap" error where longitude and latitude are inverted between mapping libraries and spatial databases, the codebase enforces strict mathematical invariants encapsulated in [`backend/app/domain/sar_intelligence/coordinates.py`](file:///c:/Users/Aayush%20Barsainya/Downloads/DL_CourseProject/maritime-operations-ai/backend/app/domain/sar_intelligence/coordinates.py):

#### Table 4.9: Coordinate Convention and Axis-Order Standards Across Subsystems
| Subsystem / Technology Layer | Axis Order Convention | Example Coordinate Tuple | Standard Specification Reference |
| :--- | :---: | :---: | :--- |
| **Python Domain Model (`GeoPoint`)** | `(latitude, longitude)` | `(50.123456, -1.234567)` | Application domain model |
| **GeoJSON Specifications** | `[longitude, latitude]` | `[-1.234567, 50.123456]` | RFC 7946 Standard (X, Y Order) |
| **PostgreSQL / PostGIS (EWKT)** | `POINT(longitude latitude)` | `SRID=4326;POINT(-1.234567 50.123456)` | OGC Well-Known Text (X, Y Order) |
| **PostGIS Geometry Constructor** | `ST_MakePoint(lon, lat)` | `ST_MakePoint(-1.234567, 50.123456)` | PostGIS SQL Functions |
| **Leaflet / React Map View** | `[latitude, longitude]` | `[50.123456, -1.234567]` | Leaflet L.latLng convention |
| **MapLibre GL / Mapbox GL** | `[longitude, latitude]` | `[-1.234567, 50.123456]` | MapLibre LngLatLike array |
| **Raster Pixel Matrix (`PixelPoint`)** | `(pixel_x, pixel_y)` | `(1240.0, 3450.0)` | Top-left origin $(0, 0)$, column/row |

### Mathematical Invariants & Antimeridian Normalization:
- **Latitude Invariant:** $-90.0 \le \phi \le +90.0$, strictly finite real number (NaN and $\pm\infty$ raise `CoordinateValidationError`).
- **Longitude Invariant:** $-180.0 \le \lambda < +180.0$.
- **Antimeridian Wrapping Algorithm:**
  $$\lambda_{norm} = (\lambda + 180.0) \pmod{360.0} - 180.0$$
  This formulation guarantees continuous geodesic calculations when vessels or radar swaths traverse the $180^\circ$ international date line.

---

## 3. Pixel-to-Geographic Georeferencing

### 3.1 Comparison of Transformation Methods
Spaceborne SAR swaths are subject to non-linear geometric distortions arising from orbital inclination ($98.18^\circ$), Earth oblateness (WGS84 ellipsoid), and the non-linear relationship between slant-range delay time and ground-range distance.

```
+-------------------------------------------------------------------------------------------------+
|                                 GEOREFERENCING METHODOLOGY COMPARISON                           |
+--------------------------+-------------------------------------+--------------------------------+
| Metric / Feature         | Global 4-Corner Affine Transform    | Product GCP Piecewise Bilinear |
+--------------------------+-------------------------------------+--------------------------------+
| Mathematical Model       | Planar 2D Affine Matrix (6 params)  | Piecewise Bilinear Mesh Surface|
| Distortion Handling      | Assumes flat, linear radar swath    | Accounts for Earth curvature   |
| Geometric Error (Swath)  | 250m - 1200m residual error         | <= 30m nominal residual error  |
| Swath Width Suitability  | Small chips only (< 10km)           | Full wide swaths (250km IW)    |
| Operational Status       | Rejected (Unacceptable error)       | IMPLEMENTED & VALIDATED        |
+--------------------------+-------------------------------------+--------------------------------+
```

### 3.2 Product Geolocation Grid Point (GCP) Implementation
Sentinel-1 products embed a grid of approximately $21 \times 11$ high-precision Geolocation Grid Points (GCPs) in their annotation metadata XML. Each grid point records:
$$GCP_k = \{ \text{line}_k, \text{pixel}_k, \text{latitude}_k, \text{longitude}_k, \text{height}_k \}$$

To map any arbitrary parent-scene pixel $(X, Y)$ to $(\text{Lon}, \text{Lat})$:
1. **Line Interval Bisection:** Perform binary bisection search across sorted GCP line indices to locate enclosing rows $[L_0, L_1]$:
   $$L_0 \le Y \le L_1$$
2. **Column Interval Bisection:** Perform binary bisection search along the points of $L_0$ and $L_1$ to find enclosing range pixels:
   $$P_{00} \le X \le P_{10} \quad (\text{on line } L_0), \qquad P_{01} \le X \le P_{11} \quad (\text{on line } L_1)$$
3. **Normalized Surface Weights:**
   $$u = \frac{X - P_{00}}{\max(P_{10} - P_{00}, 1.0)}, \qquad v = \frac{Y - L_0}{\max(L_1 - L_0, 1.0)}$$
4. **Piecewise Bilinear Interpolation:**
   $$\lambda(X, Y) = (1-u)(1-v)\lambda_{00} + u(1-v)\lambda_{10} + uv\lambda_{11} + (1-u)v\lambda_{01}$$
   $$\phi(X, Y) = (1-u)(1-v)\phi_{00} + u(1-v)\phi_{10} + uv\phi_{11} + (1-u)v\phi_{01}$$

---

## 4. Kinematic Uncertainty-Aware Spatiotemporal Correlation

### 4.1 Resolving the Vessel Displacement vs Geolocation Error Confusion
A foundational error in amateur surveillance design is conflating **kinematic vessel displacement** with **sensor geolocation error**:

> [!WARNING]
> **Vessel Travel is NOT Geolocation Error:**
> If a container ship travels at $20\text{ knots}$ ($10.29\,\text{m/s}$), in an 18-minute elapsed interval between an AIS broadcast and a satellite pass, the vessel physically travels **$11.11\,\text{kilometers}$**.
> 
> Labeling this $11.11\,\text{km}$ separation as "radar georeferencing error" or using a universal $11\,\text{km}$ static search circle is catastrophic:
> 1. In dense shipping lanes (e.g., Dover Strait), an $11\,\text{km}$ static search circle contains 30 to 80 distinct vessels, causing complete correlation ambiguity.
> 2. The $11.11\,\text{km}$ separation represents physical, deterministic trajectory displacement along the vessel's heading, whereas satellite geolocation uncertainty is an isotropic stochastic variance ($\sigma_{sar} \approx 30\,\text{m}$).

### 4.2 Great-Circle Kinematic Dead Reckoning
To bridge the time delta $\Delta t = T_{sat} - T_{ais}$, our correlation engine projects the candidate's reported AIS coordinate $(lat_0, lon_0)$ along its Course Over Ground ($\theta$) at Speed Over Ground ($v$):

$$\text{Displacement: } d = v \cdot \Delta t$$
$$\phi_{proj} = \arcsin\left(\sin \phi_0 \cos\left(\frac{d}{R}\right) + \cos \phi_0 \sin\left(\frac{d}{R}\right) \cos \theta\right)$$
$$\lambda_{proj} = \lambda_0 + \arctan2\left(\sin \theta \sin\left(\frac{d}{R}\right) \cos \phi_0, \; \cos\left(\frac{d}{R}\right) - \sin \phi_0 \sin \phi_{proj}\right)$$

where $R = 6,371,000\,\text{meters}$ (mean Earth radius).

### 4.3 Composite 3-Component Uncertainty Budget
Even with dead reckoning, the vessel may alter engine throttle, execute course corrections, or experience wind drift. We formulate a composite uncertainty budget $\sigma_{total}$:

$$\sigma_{total} = \sqrt{\sigma_{sar}^2 + \sigma_{ais}^2 + \sigma_{motion}^2}$$

1. **Satellite Georeferencing Uncertainty ($\sigma_{sar}$):** $\sigma_{sar} = 30.0\,\text{meters}$, representing residual GCP piecewise bilinear error and orbital ephemeris noise.
2. **AIS Position Measurement Uncertainty ($\sigma_{ais}$):** $\sigma_{ais} = 15.0\,\text{meters}$, representing marine GNSS receiver variance.
3. **Kinematic Motion & Maneuver Uncertainty ($\sigma_{motion}$):** Derived from acceleration variance, turning rate, and ocean surface drift over elapsed time $\Delta t$:
   $$\sigma_{motion} = \max\left(15.0\,\text{m}, \; 0.08 \cdot v_{mps} \cdot |\Delta t| + 0.15 \cdot |\Delta t|\right)$$

#### Empirical Uncertainty Budget Scaling (at 20 knots):
- At $\Delta t = 2\text{ minutes}$ ($120\,\text{s}$): $\sigma_{motion} = 15.0\,\text{m} \implies \sigma_{total} = 36.7\,\text{m}$
- At $\Delta t = 5\text{ minutes}$ ($300\,\text{s}$): $\sigma_{motion} = 29.2\,\text{m} \implies \sigma_{total} = 44.5\,\text{m}$
- At $\Delta t = 10\text{ minutes}$ ($600\,\text{s}$): $\sigma_{motion} = 58.4\,\text{m} \implies \sigma_{total} = 67.3\,\text{m}$
- At $\Delta t = 15\text{ minutes}$ ($900\,\text{s}$): $\sigma_{motion} = 87.6\,\text{m} \implies \sigma_{total} = 93.8\,\text{m}$

*Refer to Figure 4.6: Positional Uncertainty Budget Breakdown (`figures/fig4_6_uncertainty_budget_breakdown.svg` / `.png`).*

### 4.4 Standardized Error & Gaussian Likelihood Scoring
Using the geodesic distance $d_{proj}$ between the radar detection centroid and the dead-reckoned AIS position:

$$z = \frac{d_{proj}}{\max(\sigma_{total}, 1.0)}$$
$$\text{Likelihood Score: } S = \exp\left(-\frac{1}{2} \cdot \min(z, 50.0)^2\right)$$

- **Hard Gate:** $d_{hard} = \min(3000\,\text{m}, \max(150\,\text{m}, 2.5 \cdot \sigma_{total}))$
- **Soft Gate:** $d_{soft} = \min(4500\,\text{m}, 1.6 \cdot d_{hard})$

---

## 5. Decision Boundary Taxonomy and Outcome Classification

*Refer to Figure 3.9: Dark-Vessel Candidate Ethical Classification Flowchart (`figures/fig3_9_dark_vessel_candidate_classification.svg` / `.mmd`).*

The correlation engine maps candidates into four rigorous operational states:

1. **`AIS-MATCHED` (Verified Commercial Traffic):**
   - **Condition:** $d_{proj} \le d_{hard}$ AND $z \le 2.5$ ($S \ge 0.0439$).
   - **Operational Action:** Candidate is positively correlated with an active vessel. Logs matched MMSI, commercial name, SOG, COG, and displacement error. Interdiction alerts suppressed.
2. **`AMBIGUOUS / DETECTION-REVIEW-REQUIRED` (Soft-Edge Candidate):**
   - **Condition:** $d_{hard} < d_{proj} \le d_{soft}$ OR $2.5 < z \le 4.0$.
   - **Operational Action:** Target lies in the statistical uncertainty boundary (e.g., vessel executing sharp maneuvers or minor GPS drift). Highlighted on tactical dashboard with an amber warning ring, prompting human analyst inspection.
3. **`AIS-UNMATCHED` (Potential Dark-Vessel Candidate):**
   - **Condition:** No candidate passes gates AND regional $35\text{ km}$ sector audit confirms $\ge 1$ active AIS vessel.
   - **Operational Action:** High-priority tactical alert. Contact represents a confirmed surface vessel operating without broadcast telemetry in an active monitoring sector. Generates target dossier with length and heading estimates.
4. **`INSUFFICIENT-COVERAGE` (Sensor Blindspot):**
   - **Condition:** No candidate passes gates AND regional $35\text{ km}$ sector audit logs $0$ active AIS vessels.
   - **Operational Action:** Contact rendered with neutral gray symbology. Prevents false dark-vessel alerts in remote ocean areas lacking coastal receiver infrastructure or during satellite AIS message collision outages.
