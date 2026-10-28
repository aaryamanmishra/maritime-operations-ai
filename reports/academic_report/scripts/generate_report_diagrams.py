import os
from pathlib import Path

fig_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\maritime-operations-ai\reports\academic_report\figures")
fig_dir.mkdir(parents=True, exist_ok=True)

diagrams = {
    "fig3_1_overall_system_architecture.mmd": """graph TB
    subgraph SATELLITE_DATA["Copernicus Sentinel-1 SAR Constellation"]
        S1A["Sentinel-1A / 1C SAR"] -->|C-Band 5.405 GHz radar swath| CDSE["Copernicus Data Space Ecosystem (CDSE)"]
    end

    subgraph AIS_INGESTION["Terrestrial and Satellite AIS Stream"]
        AIS_STREAM["AISStream.io WebSocket / NMEA Hub"] -->|AIVDM/AIVDO JSON frames| WORKER["AIS Ingestion Worker (Python/AsyncIO)"]
    end

    subgraph BACKEND_PLATFORM["Maritime Operations AI Core (FastAPI / Python 3.12)"]
        ACQ["Copernicus Acquisition Adapter"]
        PREPROC["Radiometric Calibration and Native-Resolution Tiler"]
        YOLO_DET["YOLO SAR Vessel Detector (YOLO11n / YOLOv8s)"]
        GEOREF["GCP Piecewise Bilinear Georeferencer"]
        CORR["Kinematic Uncertainty Correlation Engine"]
        ROUTING["Weather Routing and TFT Vessel Performance Engine"]
    end

    subgraph PERSISTENCE["Geospatial Storage and Caching Layer"]
        POSTGIS[("PostgreSQL 16 + PostGIS 3.4<br/>Spatial GIST Indices (EPSG:4326)")]
        REDIS[("Redis 7 Alpine<br/>Geospatial Pub/Sub and Ephemeral Cache")]
    end

    subgraph OPERATIONAL_UI["Maritime Tactical Command Dashboard (React 19 / Vite)"]
        MAP["MapLibre GL Vector Maritime Map (EPSG:4326)"]
        PANEL_DARK["Dark-Vessel Tactical Control Panel"]
        PANEL_TRAFFIC["Real-Time Vessel Traffic Monitor"]
        PANEL_ROUTING["Weather-Optimal Voyage Routing Panel"]
    end

    CDSE -->|STAC API / OData Zip Download| ACQ
    ACQ --> PREPROC
    PREPROC -->|Native 640x640 Chips (Overlap=64px)| YOLO_DET
    YOLO_DET -->|Pixel Bounding Boxes and Confidences| GEOREF
    GEOREF -->|WGS84 Lon/Lat Coordinates| CORR
    WORKER -->|Batch Upsert Vessel State| POSTGIS
    WORKER -->|Stream Live Positions| REDIS
    POSTGIS -->|Historical AIS Search (+-15 min, 3km)| CORR
    CORR -->|AIS-MATCHED / AIS-UNMATCHED Candidates| POSTGIS
    POSTGIS -->|REST API (JSON GeoJSON)| OPERATIONAL_UI
    REDIS -->|WebSocket Live Updates| OPERATIONAL_UI
""",

    "fig3_2_end_to_end_data_flow.mmd": """sequenceDiagram
    autonumber
    actor Operator as Maritime Watch Officer
    participant UI as Tactical React Frontend
    participant API as FastAPI Backend Engine
    participant CDSE as Copernicus CDSE (ESA)
    participant ML as YOLO SAR Detector
    participant Geo as Georeferencing Engine
    participant DB as PostGIS Geospatial DB
    participant AIS as AISStream Live Worker

    Note over AIS,DB: Continuous AIS Background Ingestion
    AIS->>DB: Ingest vessel positions, MMSI, SOG, COG (PostGIS EPSG:4326)

    Operator->>UI: Select Area of Interest (BBox) and Time Window
    UI->>API: POST /api/v1/sar/search (BBox, Dates, Modality=GRD)
    API->>CDSE: STAC Catalog Query (Sentinel-1 GRD IW)
    CDSE-->>API: Return Scene Summaries (Orbit, Pass, Footprint)
    API-->>UI: Display Available Satellite Passes on Map

    Operator->>UI: Click 'Analyze Scene for Dark Vessels'
    UI->>API: POST /api/v1/sar/jobs (scene_id, modality)
    API-->>UI: 202 Accepted (job_id, status=PENDING)

    Note over API,CDSE: Asynchronous Background Processing
    API->>CDSE: Download VV/VH Polarized GeoTIFF Measurements
    API->>API: Percentile Clip Normalization and 640x640 Native Tiling
    API->>ML: Run Batched Inference (Ultralytics YOLO)
    ML-->>API: Raw Detections (bbox_x, bbox_y, confidence)
    API->>Geo: Interpolate GCP Grid to WGS84 (Lon, Lat)
    API->>API: Cross-Tile Deduplication and Boundary NMS

    API->>DB: Spatiotemporal Query AIS within +-15m and 3.0km
    DB-->>API: Candidate AIS Tracks (SOG, COG, Timestamp)
    API->>API: Compute Kinematic Uncertainty Budget and Likelihood
    API->>DB: Persist SARDetection and AISSARCorrelation Records

    UI->>API: GET /api/v1/sar/jobs/{job_id} (Poll Status)
    API-->>UI: 200 OK (status=COMPLETED, detections summary)
    UI->>DB: Fetch Rendered GeoJSON Layers
    UI-->>Operator: Render High-Contrast SAR Overlay, Matched Vessels and Dark Candidates
""",

    "fig3_3_sar_acquisition_preprocessing_flowchart.mmd": """flowchart TD
    Start([Sentinel-1 Scene Selected]) --> QueryCDSE[Query Copernicus Data Space Ecosystem API]
    QueryCDSE --> CheckCache{Scene Cached in Local Storage?}
    CheckCache -- Yes --> LoadLocal[Load Native GeoTIFF Measurement Bands from Disk]
    CheckCache -- No --> DownloadSAFE[Stream Zip/SAFE Archive via CDSE OData Access Token]
    DownloadSAFE --> ExtractBands[Extract Measurement Bands: VV and VH Polarizations]
    LoadLocal --> Radiometric[Radiometric Decibel Transformation and Noise Floor Mitigation]
    ExtractBands --> Radiometric

    Radiometric --> PercentileNorm["Dynamic Percentile Normalization: 2nd to 99.5th Percentile Clip"]
    PercentileNorm --> StackChans["Dual-Polarization Fusion: Channel 0=VV, Channel 1=VH, Channel 2=(VV+VH)/2"]
    StackChans --> ParseGCP[Parse Geolocation Grid Points from Product Annotation XML]
    ParseGCP --> ExtractFootprint[Extract Scene Geographic Footprint Polygon WGS84]
    ExtractFootprint --> TilingEngine["Sliding-Window Native Slicing Engine: Chip Size=640x640, Overlap=64px"]
    TilingEngine --> OutputChips([Output Preprocessed Native Radar Chips to Inference Queue])
""",

    "fig3_4_native_resolution_chip_extraction.mmd": """flowchart LR
    subgraph FULL_SCENE["Sentinel-1 Full Swath GeoTIFF (e.g. 25000 x 16500 pixels)"]
        direction TB
        SceneMap["Swath Image Array: 10m x 10m Ground Range Resolution"]
        TileGrid["Regular Overlapping Grid: Tile Width = 640 px, Tile Height = 640 px, Stride = 576 px (Overlap = 64 px)"]
    end

    subgraph SLICE_PROCESS["Chip Extraction Engine"]
        direction TB
        ExtractWindow["Extract Window at [y:y+640, x:x+640]"]
        PadCheck{"Window at Border? (w < 640 or h < 640)"}
        ZeroPad["Zero-Pad Edge with Background Ocean Clutter"]
        RecordOffset["Record Local-to-Global Coordinate Offsets: (x_offset, y_offset)"]
    end

    subgraph OUTPUT_BATCH["Batched Radar Chips for Detector"]
        Chip1["Chip (0, 0)"]
        Chip2["Chip (576, 0)"]
        ChipN["Chip (x_k, y_k)"]
    end

    SceneMap --> TileGrid
    TileGrid --> ExtractWindow
    ExtractWindow --> PadCheck
    PadCheck -- Yes --> ZeroPad --> RecordOffset
    PadCheck -- No --> RecordOffset
    RecordOffset --> Chip1 & Chip2 & ChipN
""",

    "fig3_5_ship_detection_inference_pipeline.mmd": """flowchart TD
    ChipsIn([Native 640x640 Radar Chips]) --> Backbone["YOLO Backbone: Modified CSPDarknet / C3k2 Blocks"]
    Backbone --> SPPF["SPPF Spatial Pyramid Pooling Fast (Multi-Scale Receptive Field)"]
    SPPF --> Neck["Path Aggregation PANet Neck (Multi-Scale Feature Fusion)"]
    Neck --> DecoupledHead["Decoupled Anchor-Free Detection Head"]

    DecoupledHead --> ClsBranch["Classification Branch: Vessel Probability"]
    DecoupledHead --> RegBranch["Regression Branch: Bounding Box Coordinates (xc, yc, w, h)"]

    ClsBranch & RegBranch --> RawFilter{"Confidence >= 0.25 Threshold?"}
    RawFilter -- No --> DiscardClutter([Discarded as Ocean Clutter])
    RawFilter -- Yes --> LocalNMS["Intra-Tile Non-Maximum Suppression (IoU >= 0.35)"]

    LocalNMS --> MapToScene["Transform Chip BBox to Parent Scene Coordinates: X = x + x_offset, Y = y + y_offset"]
    MapToScene --> CrossTileDeduplication{"Overlap Band Deduplication: IoU >= 0.30 or Near-Concentric IoMin >= 0.50?"}
    CrossTileDeduplication -- Duplicate --> SuppressTwin["Suppress Twin Detection, Retain Primary and Link Provenance"]
    CrossTileDeduplication -- Unique --> OperationalDet([Operational SAR Vessel Detection Candidate])
""",

    "fig3_6_pixel_to_geo_coordinate_transformation_flowchart.mmd": """flowchart TD
    PixelIn([Parent Scene Pixel Coordinate: X, Y]) --> LoadGCP[Load Geolocation Grid Points Grid: GCP line, pixel, lon, lat]
    LoadGCP --> LineBracket["Bisection Search: Find Enclosing GCP Lines [L0, L1] where L0 <= Y <= L1"]
    LineBracket --> PixelBracket["Bisection Search: Find Enclosing GCP Pixels [P0, P1] on Lines L0 and L1"]
    PixelBracket --> BilinearWeights["Compute Normalized Interpolation Weights: u = (X - P0)/(P1 - P0), v = (Y - L0)/(L1 - L0)"]

    BilinearWeights --> BilinearFormula["Bilinear Geodesic Surface Interpolation: Coord = (1-u)(1-v)G00 + u(1-v)G10 + uv G11 + (1-u)v G01"]
    BilinearFormula --> RawGeo["Calculated Raw Coordinate: Latitude, Longitude"]

    RawGeo --> InvariantCheck{"Validate Geometric Invariants: -90 <= Lat <= 90 and Finite Real?"}
    InvariantCheck -- Invalid --> RaiseError([Raise CoordinateValidationError])
    InvariantCheck -- Valid --> AntimeridianNorm["Antimeridian Normalization: Lon = (Lon + 180) % 360 - 180"]

    AntimeridianNorm --> FormatRepresentations["Format Standard Output Projections: Leaflet [lat, lon], GeoJSON/PostGIS [lon, lat], EWKT POINT"]
    FormatRepresentations --> ValidGeoPoint([Validated GeoPoint EPSG:4326])
""",

    "fig3_7_ais_ingestion_storage_architecture.mmd": """flowchart LR
    subgraph INGESTION_SOURCE["External Broadcast Source"]
        AIS_HUB["AISStream.io Secure WebSocket"]
    end

    subgraph WORKER_SERVICE["AIS Ingestion Worker"]
        direction TB
        WS_CLIENT["AsyncIO WebSocket Client"]
        DECODER["AIVDM/AIVDO JSON Message Parser"]
        FILTER["Geographic Bounding Box Filter: (49.5°N - 51.5°N, 2.0°W - 2.5°E)"]
        VALIDATOR["Kinematic Sanity Filter: (0 <= SOG <= 70 kn, 0 <= COG < 360°)"]
        BUFFER["Batch Buffer (100 msgs or 5.0 seconds)"]
    end

    subgraph CACHE_LAYER["Redis 7 Ephemeral Storage"]
        direction TB
        GEO_CACHE["GEOADD Live Vessel Coordinates"]
        EXPIRY["Vessel Expiry TTL = 3600 seconds"]
        PUBSUB["Pub/Sub Channel: live_vessel_traffic"]
    end

    subgraph RELATIONAL_DB["PostgreSQL 16 + PostGIS Spatial DB"]
        direction TB
        IDENTITY_TBL[("vessel_identity: MMSI, IMO, Name, Type, Dimensions")]
        STATE_TBL[("current_vessel_state: MMSI (PK), Timestamp, SOG, COG, Geom (Point)")]
        HIST_TBL[("vessel_position_history: ID (PK), MMSI, Timestamp, SOG, COG, Geom (Point)")]
    end

    AIS_HUB -->|Encrypted TLS Stream| WS_CLIENT
    WS_CLIENT --> DECODER --> FILTER --> VALIDATOR --> BUFFER
    BUFFER -->|Pipelined Write| GEO_CACHE & PUBSUB
    GEO_CACHE --> EXPIRY
    BUFFER -->|Async Batch Upsert| IDENTITY_TBL & STATE_TBL
    BUFFER -->|Async Append| HIST_TBL
""",

    "fig3_8_sar_ais_temporal_spatial_matching_flowchart.mmd": """flowchart TD
    DetIn([SAR Detection: lat_sar, lon_sar, T_sat]) --> TimeGate["Construct Temporal Filter Window: T_start = T_sat - 15 min, T_end = T_sat + 15 min"]
    TimeGate --> SpatialGate["PostGIS Spatial Candidate Query: ST_DWithin(geom::geography, ST_MakePoint(lon, lat), 4500m)"]

    SpatialGate --> QueryDB[("Search current_vessel_state and vessel_position_history")]
    QueryDB --> CandidatesCheck{"Any AIS Candidates Found?"}

    CandidatesCheck -- No Candidates --> SectorCoverage{"Check Sector Coverage (35km radius): Any active AIS pings in sector?"}
    SectorCoverage -- Active Sector --> DarkCandidate([Tag: AIS-UNMATCHED / Potential Dark-Vessel Candidate])
    SectorCoverage -- Blindspot --> InsufficientCoverage([Tag: INSUFFICIENT-COVERAGE / Unconfirmed Sector])

    CandidatesCheck -- Candidates Found --> LoopCandidates["For Each Candidate: Extract dt = T_sat - T_ais, SOG, COG"]
    LoopCandidates --> DeadReckon["Kinematic Dead Reckoning Projection: est_pos = Pos + SOG * dt * [cos COG, sin COG]"]
    DeadReckon --> GeodesicDist["Calculate Projected Distance: dist_m = Haversine(det, est_pos)"]

    GeodesicDist --> UncertaintyBudget["Calculate Composite Uncertainty Budget: sigma_total = sqrt(sigma_sar^2 + sigma_ais^2 + sigma_motion^2), z = dist_m / sigma_total, score = exp(-0.5 * z^2)"]

    UncertaintyBudget --> DecisionBoundary{"Decision Boundary Evaluation: dist_m <= hard_gate AND z <= 2.5?"}
    DecisionBoundary -- High Confidence Match --> MatchedStatus([Tag: AIS-MATCHED / Verified Commercial Vessel])
    DecisionBoundary -- Soft Edge --> ReviewStatus([Tag: DETECTION-REVIEW-REQUIRED / Ambiguous Soft Edge])
    DecisionBoundary -- Exceeded Gates --> RejectedCandidate["Candidate Exceeded Gates, Try Next Candidate"]
    RejectedCandidate --> SectorCoverage
""",

    "fig3_9_dark_vessel_candidate_classification_flowchart.mmd": """flowchart TD
    DetReady([SAR Vessel Detection Localized and Georeferenced]) --> SpatioTemporalMatching[Execute Spatiotemporal Kinematic Matching Engine]

    SpatioTemporalMatching --> EvalOutcome{Matching Engine Classification Outcome}

    EvalOutcome -- High Confidence Match --> Matched["AIS-MATCHED: Correlated with active AIS transponder. Record: MMSI, Name, SOG, COG, delta_t, distance"]
    EvalOutcome -- Soft-Edge Boundary --> Ambiguous["AMBIGUOUS / DETECTION-REVIEW-REQUIRED: Candidate near decision gate (z between 2.5 and 4.0). Requires tactical analyst inspection"]
    EvalOutcome -- No Match in Active Sector --> DarkVessel["AIS-UNMATCHED: Confirmed SAR target with active AIS coverage in 35km radius. Designation: Potential Dark-Vessel Candidate"]
    EvalOutcome -- No Match in Unmonitored Sector --> NoCoverage["INSUFFICIENT-COVERAGE: No AIS receivers or active vessels detected within 35km. Designation: Unconfirmed Operational Sensor Blindspot"]

    Matched --> TacticalAction1["Log Verified Traffic Track, Suppress Interdiction Alert"]
    Ambiguous --> TacticalAction2["Highlight on Tactical Map with Amber Review Ring"]
    DarkVessel --> TacticalAction3["Flag High-Priority Red Marker, Log Dark-Vessel Dossier"]
    NoCoverage --> TacticalAction4["Render Gray Marker, Note Sensor Coverage Limitation"]

    TacticalAction1 & TacticalAction2 & TacticalAction3 & TacticalAction4 --> AuditPersistence[("Persist in PostGIS: sar_detection and ais_sar_correlation")]
""",

    "fig3_10_database_er_diagram.mmd": """erDiagram
    SAR_SCENE ||--o{ SAR_DETECTION : contains
    SAR_DETECTION ||--o| AIS_SAR_CORRELATION : evaluates
    VESSEL_IDENTITY ||--o| CURRENT_VESSEL_STATE : tracks
    VESSEL_IDENTITY ||--o{ VESSEL_POSITION_HISTORY : records
    AIS_SAR_CORRELATION }o--o| VESSEL_IDENTITY : identifies

    SAR_SCENE {
        string scene_id PK "Unique CDSE Scene ID"
        timestamptz acquisition_time "Satellite pass timestamp"
        string platform "Sentinel-1A / Sentinel-1C"
        string orbit_pass "ASCENDING / DESCENDING"
        geometry footprint "POLYGON EPSG:4326"
        string processing_level "LEVEL1_GRD / LEVEL1_SLC"
        string polarization "VV+VH"
        jsonb scene_metadata "CDSE technical attributes"
    }

    SAR_DETECTION {
        bigint id PK "Auto-increment surrogate key"
        string detection_id UK "UUID string identifier"
        string scene_id FK "Parent scene foreign key"
        geometry geom "POINT EPSG:4326 GIST index"
        numeric latitude "WGS84 decimal degrees"
        numeric longitude "WGS84 decimal degrees"
        jsonb pixel_bbox "BBox in scene coordinates"
        numeric confidence "Model detection confidence"
        string model_version "Model version identifier"
        numeric length_m "Estimated vessel length"
        numeric heading_deg "Estimated target heading"
    }

    AIS_SAR_CORRELATION {
        bigint id PK "Auto-increment surrogate key"
        bigint sar_detection_id FK "Detection reference (Unique)"
        bigint candidate_mmsi FK "Correlated MMSI identifier"
        numeric time_difference_seconds "Delta t between SAR and AIS"
        numeric distance_km "Projected geodesic separation"
        string match_status "AIS-MATCHED / AIS-UNMATCHED / REVIEW"
        string matching_method "Kinematic uncertainty geodesic"
        jsonb provenance "Uncertainty budget and gate metrics"
    }

    VESSEL_IDENTITY {
        bigint mmsi PK "Maritime Mobile Service Identity"
        integer imo "International Maritime Organization ID"
        string name "Vessel commercial name"
        string callsign "Radio callsign"
        integer ship_type "Vessel type code"
        numeric length "Vessel overall length"
        numeric beam "Vessel breadth"
    }

    CURRENT_VESSEL_STATE {
        bigint mmsi PK, FK "Reference to vessel_identity"
        timestamptz timestamp "Latest position timestamp"
        geometry geom "POINT EPSG:4326 GIST index"
        numeric latitude "WGS84 decimal degrees"
        numeric longitude "WGS84 decimal degrees"
        numeric sog "Speed Over Ground (knots)"
        numeric cog "Course Over Ground (degrees)"
        numeric heading "Vessel compass heading"
    }

    VESSEL_POSITION_HISTORY {
        bigint id PK "Auto-increment surrogate key"
        bigint mmsi FK "Reference to vessel_identity"
        timestamptz timestamp "AIS message broadcast timestamp"
        geometry geom "POINT EPSG:4326 GIST index"
        numeric latitude "WGS84 decimal degrees"
        numeric longitude "WGS84 decimal degrees"
        numeric sog "Speed Over Ground (knots)"
        numeric cog "Course Over Ground (degrees)"
    }
""",

    "fig3_11_uml_use_case_diagram.mmd": """graph LR
    subgraph ACTORS["System Actors"]
        Officer["Maritime Watch Officer"]
        Analyst["Tactical Intelligence Analyst"]
        SysAdmin["System Administrator"]
        CDSE_EXT["Copernicus CDSE API"]
        AIS_EXT["AISStream Network"]
    end

    subgraph USE_CASES["Maritime Operations AI Capabilities"]
        UC1["UC-01: Monitor Live Maritime Traffic"]
        UC2["UC-02: Query Satellite SAR Passes"]
        UC3["UC-03: Trigger Native-Resolution Ship Detection"]
        UC4["UC-04: Classify Dark-Vessel Candidates"]
        UC5["UC-05: Inspect Uncertainty Budget and Provenance"]
        UC6["UC-06: Plan Weather-Optimized Route"]
        UC7["UC-07: Ingest and Cleanse Real-Time AIS Stream"]
        UC8["UC-08: Ingest Copernicus Satellite Swaths"]
        UC9["UC-09: Monitor System Health and Subsystems"]
    end

    Officer --> UC1
    Officer --> UC2
    Officer --> UC3
    Officer --> UC4
    Officer --> UC6

    Analyst --> UC4
    Analyst --> UC5
    Analyst --> UC3

    SysAdmin --> UC9

    AIS_EXT --> UC7
    CDSE_EXT --> UC8
    UC3 -.->|includes| UC8
    UC4 -.->|includes| UC7
""",

    "fig3_12_backend_frontend_interaction.mmd": """sequenceDiagram
    autonumber
    participant UI as React Map Component (MaritimeMap.tsx)
    participant State as Application State (App.tsx)
    participant DarkPanel as DarkVesselControlPanel.tsx
    participant SAR_API as SAR Client Service (sarApi.ts)
    participant Backend as FastAPI Application (/api/v1)
    participant DB as PostgreSQL / PostGIS Engine

    UI->>State: User clicks Area of Interest on Leaflet Map
    State->>DarkPanel: Update Bounding Box Coordinates (minLat, maxLat, minLon, maxLon)
    DarkPanel->>SAR_API: searchSARScenes(bbox, dates, modality)
    SAR_API->>Backend: POST /api/v1/sar/search
    Backend-->>SAR_API: 200 OK (List of SARSceneSummary)
    SAR_API-->>DarkPanel: Display Available Scenes in Table

    DarkPanel->>State: User Selects Scene S1C_IW_GRDH_...
    State->>UI: Draw Footprint Polygon on Map

    DarkPanel->>SAR_API: startSARJob(scene_id, modality)
    SAR_API->>Backend: POST /api/v1/sar/jobs
    Backend-->>SAR_API: 202 Accepted (job_id, status=PENDING)

    loop Polling every 2.5s until COMPLETED
        DarkPanel->>SAR_API: fetchSARJobStatus(job_id)
        SAR_API->>Backend: GET /api/v1/sar/jobs/{job_id}
        Backend-->>SAR_API: Job Status (PROGRESS / COMPLETED)
    end

    DarkPanel->>SAR_API: fetchSceneDetections(scene_id)
    SAR_API->>Backend: GET /api/v1/sar/scenes/{scene_id}/detections
    Backend->>DB: Query SARDetection with AISSARCorrelation join
    DB-->>Backend: Detections Records with Provenance JSON
    Backend-->>SAR_API: 200 OK (List of SARDetectionItem)
    SAR_API-->>DarkPanel: Load Detections into List
    DarkPanel->>State: setSARDetections(detections)
    State->>UI: Render High-Contrast Overlays (Red=Dark, Green=Matched, Amber=Review)
"""
}

for name, content in diagrams.items():
    target = fig_dir / name
    target.write_text(content.strip(), encoding="utf-8")
    print(f"Generated {target.name}")

# Also copy to root reports dir
root_fig_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\reports\academic_report\figures")
root_fig_dir.mkdir(parents=True, exist_ok=True)
for name, content in diagrams.items():
    (root_fig_dir / name).write_text(content.strip(), encoding="utf-8")

print("All 12 Mermaid source files written successfully to both locations.")
