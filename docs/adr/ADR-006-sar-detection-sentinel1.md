# ADR-006: Sentinel-1 SAR Vessel Detection and AIS Spatial Correlation

## Status
Accepted

## Context
Non-broadcasting maritime contacts ("dark vessels") do not transmit AIS or disable transponders. Satellite Synthetic Aperture Radar (SAR) operates day and night and penetrates cloud cover, making it the premier open remote sensing modality for ocean surface surveillance.

Key challenges include:
- SAR scenes (e.g. Copernicus Sentinel-1 Level-1 GRD) are large files (1 to 1.5 GB per scene).
- Pixel coordinates must be orthorectified and geocoded to WGS84 coordinates.
- Traditional thresholding (CFAR) generates false positives from ocean clutter, sea ice, and azimuth ambiguities.
- Satellite passes provide discrete snapshot observations at a specific UTC timestamp, not continuous real-time video or tracking.
- Non-broadcasting contacts must be treated objectively: failure to match AIS does not automatically denote illegal activity (small craft, military vessels, technical transmission outages, and coastal topology shadowing are common legitimate causes).

## Decision
1. **Target Modality**: Copernicus Sentinel-1 C-band SAR Level-1 Ground Range Detected (GRD) products in IW (Interferometric Wide swath) mode, acquired via the free Copernicus Data Space Ecosystem (CDSE) STAC API and OData service.
2. **Inference Pipeline**:
   - Automated ROI slicing: Scenes are dynamically sliced into $800 \times 800$ pixel chips focusing on open ocean water masks (excluding land using GSHHG coastline masks to prevent false land positives).
   - Deep Learning Detector: A lightweight object detection model (YOLOv8-SAR or Faster R-CNN) trained on established open SAR ship benchmarks (LS-SSDD-v1.0 / HRSID / xView3-SAR). A classic 2-parameter Constant False Alarm Rate (CFAR) algorithm is provided as an interpretable fallback baseline.
3. **AIS Spatiotemporal Correlation Engine**:
   - For each detected target bounding box center $(lat_{det}, lon_{det})$ at satellite acquisition time $T_{sat}$, the engine queries PostGIS for all AIS vessel positions recorded within a spatial radius ($R = 3.0\text{ km}$, accounting for target velocity and georeferencing error) and temporal window $[T_{sat} - 15\text{ min}, T_{sat} + 15\text{ min}]$.
   - Targets with matching interpolated AIS records are classified as `MATCHED_AIS`.
   - Targets without matching AIS records are classified as `UNMATCHED_AIS_CONTACT`.
4. **Terminology & Ethics Guardrails**:
   - The UI and API strictly label unmatched detections as **"AIS-Unmatched Contacts"** or **"Non-Reporting Radar Contacts"**.
   - The term "illegal vessel" is strictly prohibited across all schemas, logs, and frontend views.
   - Each detection explicitly displays its satellite acquisition timestamp and sensor geometry.

## Consequences
### Positive
- Fully reproducible using open satellite data without commercial radar imagery costs.
- High ethical and scientific rigor in user-facing dark-vessel intelligence.
- Accurate historical correlation using PostGIS spatial indexing.

### Negative / Trade-offs
- Free Copernicus download speeds and quotas can introduce latency during full-scene processing; localized chip caching is required.
