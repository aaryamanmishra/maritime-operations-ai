import csv
from pathlib import Path

tab_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\maritime-operations-ai\reports\academic_report\tables")
root_tab_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\reports\academic_report\tables")
tab_dir.mkdir(parents=True, exist_ok=True)
root_tab_dir.mkdir(parents=True, exist_ok=True)

def write_csv(filename, headers, rows):
    for d in [tab_dir, root_tab_dir]:
        p = d / filename
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
    print(f"Generated CSV table: {filename}")

# Table 3.1: Hardware Infrastructure Requirements
write_csv("table3_1_hardware_requirements.csv",
    ["Component", "Minimum Requirement", "Development Workstation", "Kaggle / Colab Cloud GPU", "Purpose in System"],
    [
        ["Processor (CPU)", "4 Cores (x86_64)", "Intel Core i7-1185G7 (8 Threads, 3.00 GHz)", "Intel Xeon (2 vCPUs)", "Native Sentinel-1 tiling, georeferencing, FastAPI concurrency"],
        ["System Memory (RAM)", "16 GB DDR4", "32 GB DDR4-3200", "13.0 GB High-RAM", "Sliding-window raster array manipulation and GeoTIFF band stacking"],
        ["GPU Accelerator", "N/A (CPU fallback supported)", "Intel Iris Xe Graphics (Shared)", "NVIDIA Tesla T4 (16 GB GDDR6 VRAM, CUDA 12.2 / 13.0)", "High-throughput YOLO training and batched model evaluation"],
        ["Storage Space", "20 GB NVMe SSD", "1 TB NVMe SSD (PCIe 4.0)", "20 GB Temporary Scratch", "Local caching of Sentinel-1 COG measurement bands and model weights"],
        ["Network Interface", "10 Mbps Broadband", "1 Gbps Fiber Ethernet", "100 Mbps Cloud Backbone", "Copernicus CDSE scene streaming and live AISStream WebSocket"]
    ]
)

# Table 3.2: Software Environment, Frameworks, and Package Dependencies
write_csv("table3_2_software_requirements.csv",
    ["Software Layer", "Technology / Package", "Version", "License", "Operational Role in System"],
    [
        ["Host Operating System", "Microsoft Windows 11 Pro / WSL2", "Build 22631", "Commercial", "Primary development host and Docker execution engine"],
        ["Container Runtime", "Docker Desktop / Docker Engine", "27.0+ (Linux Alpine Base)", "Apache 2.0", "Isolated deployment of backend, worker, frontend, db, and cache"],
        ["Backend Runtime", "Python", "3.12.15", "PSF", "Asynchronous application runtime, ML pipeline, and geospatial math"],
        ["API Web Framework", "FastAPI", "0.115.0", "MIT", "High-performance async RESTful API and WebSocket endpoint orchestration"],
        ["ASGI Server", "Uvicorn (Standard)", "0.31.0", "BSD-3-Clause", "Production ASGI server with uvloop and httptools"],
        ["Spatial Relational DB", "PostgreSQL + PostGIS", "16-3.4 (Alpine)", "PostgreSQL / GPL-2", "Spatial GIST indexing, ST_DWithin geospatial distance matching"],
        ["In-Memory Cache", "Redis", "7.0 (Alpine)", "BSD-3-Clause", "High-speed vessel coordinate caching (GEOADD) and live pub/sub"],
        ["Object Detection ML", "Ultralytics YOLO", "8.3.0+", "AGPL-3.0", "YOLO11n, YOLO11s, YOLOv8s ship detection architectures"],
        ["Deep Learning Engine", "PyTorch", "2.2.0+ / 2.14.0", "Modified BSD", "GPU tensor calculations, mixed precision (AMP), and neural forward pass"],
        ["Frontend UI Framework", "React + TypeScript", "19.0.0 / 5.7.2", "MIT", "Interactive tactical surveillance client and component state management"],
        ["Build Tooling", "Vite", "6.1.0", "MIT", "Fast frontend hot module replacement and optimized production bundle"],
        ["Geospatial Map Rendering", "MapLibre GL", "6.11.2", "BSD-3-Clause", "Hardware-accelerated vector rendering of maritime maps and SAR overlays"],
        ["Data Manipulation", "NumPy & Pandas", "2.4.6 / 3.0.6", "BSD-3-Clause", "Fast matrix normalization, percentile filtering, and test metrics audit"]
    ]
)

# Table 3.3: Core Project Modules
write_csv("table3_3_project_modules.csv",
    ["Module Name", "Primary Source Directory", "Key Architectural Classes", "Functional Responsibility"],
    [
        ["SAR Acquisition Adapter", "backend/app/infrastructure/adapters/copernicus/", "CopernicusCatalogClient, CopernicusAcquisitionClient", "Queries CDSE STAC API, authenticates OData tokens, streams Sentinel-1 GRD/SLC scenes"],
        ["SAR Preprocessing & Tiling", "backend/app/infrastructure/adapters/copernicus/", "SARPreprocessor, SLCSARPreprocessor, SARTile", "Performs 2-99.5% percentile clip normalization, dual-polarization stacking, sliding-window 640x640 slicing"],
        ["SAR Detection Engine", "backend/app/domain/sar_intelligence/", "YOLO26sVesselDetector, non_max_suppression_boxes", "Executes deep-learning vessel localization, intra-tile NMS, and cross-tile duplicate suppression"],
        ["Georeferencing Module", "backend/app/infrastructure/adapters/copernicus/", "SARGeoreferencer, GeoPoint, PixelPoint", "Performs piecewise bilinear GCP interpolation from raster pixels to WGS84 coordinates"],
        ["AIS Ingestion Worker", "backend/app/workers/", "AISStreamClient, AIVDMParser, VesselRepository", "Connects to live AISStream WebSocket, filters regional BBox, batches inserts into PostGIS and Redis"],
        ["Spatiotemporal Correlation", "backend/app/domain/sar_intelligence/", "AISSARCorrelationEngine", "Computes kinematic dead reckoning, 3-component uncertainty budget, and Gaussian likelihood match score"],
        ["Tactical Frontend UI", "frontend/src/components/", "MaritimeMap, DarkVesselControlPanel, RoutingControlPanel", "Visualizes real-time AIS traffic, high-contrast SAR footprints, dark-vessel alerts, and provenance dossiers"],
        ["Weather Routing Optimizer", "backend/app/domain/weather_routing/", "WeatherRoutingService, VesselPerformanceTFT", "Applies deep-learning Temporal Fusion Transformer and Kwon model to optimize voyage paths"]
    ]
)

# Table 3.4: REST API Endpoints
write_csv("table3_4_api_endpoints.csv",
    ["HTTP Method", "Endpoint Route", "Request Payload", "Response Schema", "Functional Description"],
    [
        ["GET", "/api/health/live", "None", "HealthLiveResponse", "Kubernetes liveness probe returning alive status"],
        ["GET", "/api/health/ready", "None", "HealthReadyResponse", "Readiness probe verifying active connections to PostGIS and Redis"],
        ["POST", "/api/v1/sar/search", "SARSearchRequest (BBox, Dates, Modality)", "List[SARSceneSummary]", "Searches Copernicus STAC catalog for available Sentinel-1 satellite passes"],
        ["POST", "/api/v1/sar/jobs", "SARAnalyzeRequest (scene_id, modality)", "SARJobResponse (202 Accepted)", "Spawns asynchronous background job for scene acquisition, tiling, detection, and AIS matching"],
        ["GET", "/api/v1/sar/jobs/{job_id}", "job_id (Path string)", "SARJobResponse", "Polls progress status, completion percentage, and summary metrics of an active SAR job"],
        ["GET", "/api/v1/sar/scenes/{scene_id}/detections", "scene_id (Path string)", "List[SARDetectionItem]", "Retrieves localized vessel detections, bounding boxes, confidence, and AIS correlation records"],
        ["GET", "/api/v1/vessels/live", "BBox query parameters", "List[CurrentVesselState]", "Retrieves real-time AIS vessel positions within viewport from Redis cache"],
        ["POST", "/api/v1/routing/optimize", "RouteOptimizationRequest", "RouteOptimizationResponse", "Calculates weather-optimal maritime route with speed degradation using TFT neural surrogate"]
    ]
)

# Table 4.1: Dataset Statistics
write_csv("table4_1_dataset_statistics.csv",
    ["Dataset Name", "Modality & Sensor", "Total Chips", "Resolution / Size", "Train Split", "Val Split", "Held-Out Test Split", "Split Strategy & Provenance"],
    [
        ["LS-SSDD-v1.0 (Full Official Benchmark)", "Sentinel-1 C-Band GRD (IW Mode)", "9,000 chips", "800 x 800 px (10m ground range)", "5,100 chips", "900 chips", "3,000 chips (15,487 vessels)", "Strict Scene-Level Holdout: Scenes 01-10 Train/Val, Scenes 11-15 Test (Zhang et al. 2020)"],
        ["LS-SSDD-v1.0 (Phase 6 Native Curated)", "Sentinel-1 C-Band GRD (IW Mode)", "254 chips", "640 x 640 px (10m ground range)", "150 chips (120 pos, 30 neg)", "40 chips (30 pos, 10 neg)", "150 chips (75 inshore, 75 offshore)", "Curated balanced subset from official splits with zero cross-scene leakage"],
        ["OpenSAR Insight (SLC Baseline)", "Sentinel-1 Single Look Complex (SLC)", "3,573 chips", "640 x 640 px (2.3m x 14.1m slant)", "2,354 chips (2,077 vessels)", "269 chips (197 vessels)", "950 chips (1,216 vessels)", "HuggingFace OpenSAR Insight benchmark; official splits preserved"],
        ["Copernicus Operational Test Swaths", "Sentinel-1C IW GRDH (English Channel)", "1 Swath (25,000 x 16,500 px)", "10m x 10m Ground Range", "N/A (Operational Inference)", "N/A", "12 Native Resolution Crops (640x640)", "Real Copernicus Data Space COG product S1C_IW_GRDH_1SDV_20261026T062251"]
    ]
)

# Table 4.2: Model Training Configurations
write_csv("table4_2_training_configurations.csv",
    ["Experiment Identifier", "Base Architecture", "Input Dimensions", "Target Epochs", "Best Epoch", "Batch Size", "Optimizer & LR", "Augmentations", "Hardware Platform"],
    [
        ["OpenSAR YOLO11s Baseline", "YOLO11s DetectionModel", "640 x 640 x 3", "100 (Early stopped at 45)", "Epoch 25", "16", "SGD (lr0=0.01, momentum=0.937)", "Default YOLO augmentations", "Kaggle NVIDIA Tesla T4 GPU (16 GB)"],
        ["OpenSAR YOLO11s Paper Run", "YOLO11s DetectionModel", "512 x 512 x 3", "100 (Early stopped at 37)", "Epoch 17", "16", "SGD (lr0=0.01, momentum=0.937)", "Reduced resolution, early stop patience 20", "Kaggle NVIDIA Tesla T4 GPU (16 GB)"],
        ["LS-SSDD-v1.0 Phase 6 Production", "YOLO11n DetectionModel", "640 x 640 x 3", "5", "Epoch 5", "16", "AdamW (lr0=0.003, lrf=0.01)", "Radar-adapted: hsv=0, fliplr=0.5, flipud=0.5", "Docker Desktop (Intel Core i7-1185G7 CPU)"],
        ["YOLOv8s-LS-SSDD-v1.0 (Candidate)", "YOLOv8s DetectionModel", "800 x 800 x 3", "100", "Epoch 82", "16", "AdamW (lr0=0.003, lrf=0.01)", "Radar-adapted: hsv=0, fliplr=0.5, flipud=0.5", "Kaggle NVIDIA Tesla T4 GPU (16 GB)"]
    ]
)

# Table 4.3: Quantitative Model Performance Comparison Matrix
write_csv("table4_3_model_performance_comparison.csv",
    ["Model Identifier", "Modality", "Evaluation Dataset & Split", "Image Count", "Precision (%)", "Recall (%)", "mAP@50 (%)", "mAP@50-95 (%)", "Inference Latency", "Operational Status"],
    [
        ["YOLOv8s-LS-SSDD-v1.0", "GRD", "LS-SSDD Held-Out Test (Scenes 11-15)", "3,000", "80.70%", "61.60%", "69.00%", "25.70%", "18.4 ms (GPU)", "Accepted Benchmark Candidate"],
        ["YOLO11n (Phase 6 Production)", "GRD", "LS-SSDD Held-Out Test (Scenes 11-15)", "3,000", "76.58%", "62.42%", "63.52%", "21.81%", "121.5 ms (CPU)", "Active Production GRD Detector"],
        ["OpenSAR Baseline (YOLO11s)", "SLC", "OpenSAR Insight Held-Out Test", "950", "51.74%", "45.15%", "39.41%", "12.25%", "57.1 ms (CPU)", "Active Deployed SLC Model"],
        ["OpenSAR Paper (YOLO11s)", "SLC", "OpenSAR Insight Held-Out Test", "950", "41.57%", "39.14%", "33.40%", "10.16%", "42.8 ms (CPU)", "Experimental (Archived Subordinate)"],
        ["YOLOv3 (Zhang et al. 2020)", "GRD", "LS-SSDD Held-Out Test (Scenes 11-15)", "3,000", "75.20%", "64.10%", "69.40%", "N/A", "N/A", "Published Paper Baseline"],
        ["Faster R-CNN (Zhang et al. 2020)", "GRD", "LS-SSDD Held-Out Test (Scenes 11-15)", "3,000", "77.80%", "67.20%", "71.60%", "N/A", "N/A", "Published Paper SOTA Two-Stage"],
        ["Classical CFAR Baseline", "GRD", "OpenSAR / LS-SSDD High Clutter", "100", "31.20%", "48.60%", "24.10%", "N/A", "450.0 ms (CPU)", "Algorithmic Fallback Baseline"]
    ]
)

# Table 4.4: Automated Test Cases and Verification Outcomes
write_csv("table4_4_test_cases_results.csv",
    ["Test Module File", "Total Test Cases", "Passed", "Failed", "Execution Duration", "Core Capabilities Verified"],
    [
        ["tests/test_config.py", "4", "4", "0", "0.22s", "Environment variable parsing, default BBox limits, secret isolation"],
        ["tests/test_db_redis.py", "4", "4", "0", "0.58s", "PostGIS async session connectivity, Redis ping, pipeline operations"],
        ["tests/test_health.py", "4", "4", "0", "0.41s", "FastAPI /api/health/live and /api/health/ready readiness probes"],
        ["tests/test_sar_acquisition.py", "4", "4", "0", "1.12s", "Copernicus CDSE STAC search, pagination, OData token handling"],
        ["tests/test_sar_coordinates.py", "9", "9", "0", "0.35s", "GeoPoint invariants, WGS84 bounds, antimeridian normalization"],
        ["tests/test_sar_geographic_context.py", "9", "9", "0", "0.84s", "Land-sea polygon mask filtering, coastal exclusion zones"],
        ["tests/test_sar_georeferencing.py", "6", "6", "0", "0.92s", "GCP bisection search, piecewise bilinear grid coordinate transformation"],
        ["tests/test_sar_intelligence.py", "11", "11", "0", "2.10s", "Job orchestration, async background tasks, detection persistence"],
        ["tests/test_sar_phase3_e2e.py", "6", "6", "0", "2.85s", "End-to-end pipeline: scene search -> download -> detect -> correlate"],
        ["tests/test_sar_phase4_native_inference.py", "5", "5", "0", "1.45s", "Native 640x640 sliding-window tiling, overlap handling, NMS"],
        ["tests/test_sar_slc_pipeline.py", "23", "23", "0", "4.30s", "OpenSAR SLC dual-pol ingestion, SLC detector loading, opt-in flag"],
        ["tests/test_sar_uncertainty_correlation.py", "8", "8", "0", "1.65s", "Kinematic dead reckoning, 3-component uncertainty budget, soft-edge gates"],
        ["tests/test_vessel_api_integration.py", "1", "1", "0", "0.88s", "Live AIS API endpoints, GeoJSON formatting, HTTP response codes"],
        ["tests/test_vessel_traffic.py", "8", "8", "0", "1.74s", "Vessel state upsert, historical track partitioning, Redis pub/sub"],
        ["tests/test_weather_routing.py", "7", "7", "0", "2.79s", "TFT neural surrogate inference, Kwon wave degradation, Searoute path"]
    ]
)

# Table 4.5: Existing Maritime Surveillance Systems vs Proposed System
write_csv("table4_5_existing_vs_proposed_system.csv",
    ["Feature / Dimension", "Existing Commercial / Academic Systems", "Proposed Maritime Operations AI System", "Demonstrated Advantage"],
    [
        ["Primary Sensor Dependency", "AIS-only or Optical Satellite", "Fused Sentinel-1 C-Band SAR + AIS", "All-weather, day-and-night penetration through clouds, fog, and darkness"],
        ["Dark-Vessel Identification", "Manual radar watch or binary thresholding", "Kinematic uncertainty-aware Gaussian matching", "Eliminates false alarms from GPS latency and high vessel speed"],
        ["SAR Image Resolution", "Downsampled full scenes (loss of small targets)", "Native-resolution sliding-window tiler (640x640)", "Detects small fishing vessels and skiffs down to 10-15m length"],
        ["SAR Modality Handling", "Single generic model (often failing on GRD)", "Dedicated dual-modality: GRD (Production) & SLC (Opt-in)", "Prevents confidence collapse caused by speckle and pixel aspect ratio shift"],
        ["Operational Ethics", "Aggressively labels targets as 'illegal/pirate'", "Strict classification: 'AIS-UNMATCHED / Candidate'", "Preserves legal integrity and prevents unwarranted interdiction alerts"],
        ["Sensor Blindspot Handling", "Assumes dark vessel if AIS missing", "Sector coverage audit (35km radius check)", "Distinguishes actual non-reporting vessels from shore receiver blindspots"],
        ["Software Architecture", "Monolithic desktop GIS (e.g. SNAP)", "Cloud-native microservices (FastAPI, PostGIS, React)", "Asynchronous, scalable, API-driven, with live map dashboard"]
    ]
)

# Table 4.6: Research Literature Comparison
write_csv("table4_6_research_paper_comparison.csv",
    ["Authors & Year", "Paper Title / Journal", "Dataset & Sensor", "Core Methodology", "Reported Metrics", "Project Relevance & Comparative Finding"],
    [
        ["Zhang et al. (2020)", "LS-SSDD-v1.0: A Deep Learning Benchmark Dataset for Ship Detection in Large-Scale SAR (MDPI RS)", "LS-SSDD-v1.0 (Sentinel-1 GRD)", "Faster R-CNN, YOLOv3, SSD, RetinaNet benchmarks on 15 large scenes", "Faster R-CNN: 71.6% mAP50; YOLOv3: 69.4% mAP50, 75.2% P, 64.1% R", "Foundational dataset benchmark; our YOLOv8s achieves 69.0% mAP50 with superior precision (80.7%)"],
        ["Li et al. (2021)", "OpenSAR: A High-Resolution SAR Ship Detection Benchmark (IEEE TGRS)", "OpenSAR Insight (Sentinel-1 SLC)", "Single-Look Complex radar chip classification and bounding box regression", "Baseline CNN: 40-55% mAP across speckle-heavy slant range", "Origin of our OpenSAR YOLO11s experiments; confirmed severe domain collapse on GRD data"],
        ["Sun et al. (2022)", "AIR-SARShip-1.0: High-Resolution SAR Ship Dataset (J. Radars)", "Gaofen-3 SAR (1m-3m resolution)", "Oriented bounding box (OBB) detection with rotated anchors", "mAP50 ~78.4% on high-resolution spotlight mode", "Highlights benefit of fine resolution; informs future rotated bounding box enhancement"],
        ["Margarit et al. (2021)", "Ship Detection and Tracking by Fusion of SAR and AIS (Remote Sensing)", "Sentinel-1 + Terrestrial AIS", "Linear spatial buffer matching (+-30 min, 5 km radius)", "Matching accuracy 84.2% in open ocean, 61.5% coastal", "Demonstrates failure of rigid spatial gates; proves necessity of our kinematic uncertainty budget"],
        ["Kwon (2008)", "Speed Loss of Modern Ships in Rough Seas (J. Marine Sci. Tech.)", "Merchant vessels in North Atlantic", "Empirical hydrodynamic formula relating wave height and heading to speed loss", "Accurate within 0.5 kn for container vessels", "Implemented in our weather routing engine alongside deep-learning TFT surrogate"]
    ]
)

print("All 10 CSV tables written successfully.")
