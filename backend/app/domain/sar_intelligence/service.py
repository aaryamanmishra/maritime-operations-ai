import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np
from fastapi import BackgroundTasks
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.domain.sar_intelligence.correlation import AISSARCorrelationEngine
from app.domain.sar_intelligence.geographic_context import GeographicContextValidator
from app.domain.sar_intelligence.inference import YOLO26sVesselDetector
from app.domain.sar_intelligence.models import (
    MatchStatus,
    SARDetectionItem,
    SARJobResponse,
    SARJobStatus,
    SARModality,
    SARSceneSummary,
)
from app.infrastructure.adapters.copernicus.acquisition import (
    CopernicusAcquisitionClient,
)
from app.infrastructure.adapters.copernicus.catalog import CopernicusCatalogClient
from app.infrastructure.adapters.copernicus.georeference import SARGeoreferencer
from app.infrastructure.adapters.copernicus.preprocessing import (
    SARPreprocessor,
    SLCSARPreprocessor,
)
from app.infrastructure.database.models.sar import (
    AISSARCorrelation,
    SARDetection,
    SARScene,
)

logger = logging.getLogger(__name__)


class SARAnalysisService:
    """
    Central operational service orchestrating:
    1. STAC scene discovery
    2. Asynchronous job execution
    3. Asset acquisition & preprocessing
    4. YOLO26s vessel detection
    5. PostGIS persistence & spatiotemporal AIS correlation
    """

    _instance: Optional["SARAnalysisService"] = None

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.catalog_client = CopernicusCatalogClient(self.settings)
        self.acquisition_client = CopernicusAcquisitionClient(self.settings)
        self.preprocessor = SARPreprocessor(tile_size=640, tile_overlap=64)
        self.slc_preprocessor = SLCSARPreprocessor(tile_size=640, tile_overlap=64)
        self.correlation_engine = AISSARCorrelationEngine(
            time_window_minutes=self.settings.SAR_CORRELATION_TIME_WINDOW_MINUTES,
            max_distance_km=self.settings.SAR_CORRELATION_DISTANCE_KM,
        )
        self.geographic_validator = GeographicContextValidator.get_instance()
        self._jobs: dict[str, SARJobResponse] = {}

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> "SARAnalysisService":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    async def search_scenes(
        self,
        bbox: tuple[float, float, float, float],
        start_time: datetime,
        end_time: datetime,
        limit: int = 20,
        modality: str = "GRD",
    ) -> list[SARSceneSummary]:
        """Search Copernicus catalog for Sentinel-1 scenes (GRD default, SLC opt-in)."""
        return await self.catalog_client.search_scenes(
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            modality=modality,
        )

    def get_job(self, job_id: str) -> SARJobResponse | None:
        """Retrieve live status of an asynchronous SAR job."""
        return self._jobs.get(job_id)

    async def start_job(
        self,
        scene_id: str,
        background_tasks: BackgroundTasks,
        session_factory,
        scene_summary: SARSceneSummary | None = None,
        modality: str = "GRD",
    ) -> SARJobResponse:
        """
        Register and initiate a background SAR analysis job.
        Heavy inference is decoupled from the HTTP request cycle.
        Supports GRD (default production) or SLC (opt-in) modality pipelines.
        """
        mod_upper = modality.upper()
        # Check if an existing in-flight job for this scene and modality is already executing
        for existing_job in self._jobs.values():
            if existing_job.scene_id == scene_id and getattr(existing_job, "modality", "GRD") == mod_upper and existing_job.status in (
                SARJobStatus.QUEUED,
                SARJobStatus.DOWNLOADING,
                SARJobStatus.PREPROCESSING,
                SARJobStatus.INFERENCE,
                SARJobStatus.CORRELATING,
            ):
                logger.info("Reusing existing in-flight SAR %s job %s for scene %s", mod_upper, existing_job.job_id, scene_id)
                return existing_job

        job_id = f"sar-job-{uuid.uuid4().hex[:10]}"
        initial_job = SARJobResponse(
            job_id=job_id,
            scene_id=scene_id,
            status=SARJobStatus.QUEUED,
            modality=SARModality(mod_upper),
            progress_percent=5,
            message=f"SAR {mod_upper} analysis job queued",
        )
        self._jobs[job_id] = initial_job

        # Launch background execution
        background_tasks.add_task(
            self._execute_sar_workflow,
            job_id=job_id,
            scene_id=scene_id,
            session_factory=session_factory,
            scene_summary=scene_summary,
            modality=mod_upper,
        )

        return initial_job

    async def _execute_sar_workflow(
        self,
        job_id: str,
        scene_id: str,
        session_factory,
        scene_summary: SARSceneSummary | None = None,
        modality: str = "GRD",
    ):
        """Asynchronous execution lifecycle for SAR asset downloading, inference, and correlation."""
        job = self._jobs[job_id]
        mod_upper = modality.upper()
        logger.info("Starting SAR %s workflow for job %s (scene %s)", mod_upper, job_id, scene_id)

        try:
            # 1. DOWNLOADING PHASE
            job.status = SARJobStatus.DOWNLOADING
            job.progress_percent = 15
            job.message = "Verifying scene metadata and acquiring SAR data"

            # Check if scene metadata already exists in DB
            db_scene = None
            async with session_factory() as session:
                res = await session.execute(
                    select(SARScene).where(SARScene.scene_id == scene_id)
                )
                db_scene = res.scalar_one_or_none()

            # If not in DB or DB record lacks S3 assets, resolve metadata
            need_metadata = (
                not db_scene
                or not db_scene.scene_metadata
                or ("s3_vv" not in db_scene.scene_metadata and "s3_xml" not in db_scene.scene_metadata)
            )
            if need_metadata:
                if not scene_summary:
                    scene_summary = await self.catalog_client.get_scene_by_id(scene_id, modality=mod_upper)

                if not scene_summary:
                    # Construct fallback scene summary from ID
                    scene_summary = SARSceneSummary(
                        scene_id=scene_id,
                        acquisition_time=datetime.now(UTC),
                        platform="Sentinel-1A",
                        footprint={
                            "type": "Polygon",
                            "coordinates": [
                                [[-0.5, 50.0], [1.5, 50.2], [1.3, 50.9], [-0.7, 50.7], [-0.5, 50.0]]
                            ],
                        },
                        polarization="VV+VH",
                        processing_level="LEVEL1_GRD",
                        source="Copernicus CDSE",
                    )

                # Persist or update scene in PostgreSQL
                async with session_factory() as session:
                    footprint_wkt = (
                        f"SRID=4326;POLYGON(({', '.join(f'{pt[0]} {pt[1]}' for pt in scene_summary.footprint['coordinates'][0])}))"
                    )
                    if db_scene:
                        db_scene.scene_metadata = dict(scene_summary.scene_metadata or {})
                        session.add(db_scene)
                    else:
                        new_scene = SARScene(
                            scene_id=scene_summary.scene_id,
                            acquisition_time=scene_summary.acquisition_time,
                            platform=scene_summary.platform,
                            orbit_pass=scene_summary.orbit_pass,
                            orbit_number=scene_summary.orbit_number,
                            footprint=footprint_wkt,
                            processing_level=scene_summary.processing_level,
                            polarization=scene_summary.polarization,
                            source=scene_summary.source,
                            scene_metadata=scene_summary.scene_metadata,
                        )
                        session.add(new_scene)
                        db_scene = new_scene
                    await session.commit()

            # 2. PREPROCESSING PHASE
            job.status = SARJobStatus.PREPROCESSING
            job.progress_percent = 35
            job.message = f"Radiometric calibration, {mod_upper} normalization, and spatial tiling"

            # Check if local raster chip exists or create realistic radar raster for the scene
            cache_dir = f"data/sar_cache/{mod_upper.lower()}/{scene_id}"
            os.makedirs(cache_dir, exist_ok=True)
            raster_path = os.path.join(cache_dir, "scene_fused.png")

            effective_metadata: dict[str, Any] = {}
            if scene_summary and scene_summary.scene_metadata:
                effective_metadata.update(scene_summary.scene_metadata)
            elif db_scene and db_scene.scene_metadata:
                effective_metadata.update(db_scene.scene_metadata)

            burst_meta: dict[str, Any] = {}
            if mod_upper == "SLC":
                # Real SLC burst acquisition fails explicitly if credentials, GDAL, or data are missing.
                burst_data = await self.acquisition_client.acquire_slc_burst(
                    scene_id=scene_id,
                    target_dir=f"data/sar_cache/{mod_upper.lower()}",
                    scene_metadata=effective_metadata,
                )
                r_path = burst_data.get("raster_path")
                if r_path and os.path.exists(r_path) and os.path.getsize(r_path) > 0:
                    raster_path = r_path
                else:
                    raise RuntimeError(f"SLC burst acquisition produced no valid raster data for scene {scene_id}")

                burst_meta = burst_data.get("metadata", {})
                if not burst_meta:
                    raise RuntimeError(f"SLC burst acquisition produced no valid metadata for scene {scene_id}")

                effective_metadata.update(burst_meta)
                if scene_summary:
                    scene_summary.scene_metadata.update(burst_meta)

            is_real_scene = bool(scene_id and scene_id.startswith("S1") and "DEMO" not in scene_id.upper())
            if not os.path.exists(raster_path) and mod_upper == "GRD" and is_real_scene:
                grd_data = await self.acquisition_client.acquire_grd_scene(
                    scene_id=scene_id,
                    scene_metadata=effective_metadata,
                    target_dir=f"data/sar_cache/{mod_upper.lower()}",
                )
                r_path = grd_data.get("raster_path")
                if r_path and os.path.exists(r_path) and os.path.getsize(r_path) > 0:
                    raster_path = r_path

            # Strictly separate fallback sample images by modality (offline demo fallback only)
            is_real_scene = bool(scene_id and scene_id.startswith("S1") and "DEMO" not in scene_id.upper())
            if not os.path.exists(raster_path):
                if is_real_scene:
                    raise RuntimeError(
                        f"{mod_upper} raster data unavailable for real observation {scene_id}: "
                        "cannot execute vessel detection on missing or synthetic SAR data"
                    )
                elif mod_upper == "SLC":
                    raise RuntimeError(
                        f"SLC raster data unavailable for {scene_id}: cannot execute vessel detection on missing SAR data"
                    )
                else:
                    candidate_dirs = ["data/processed/opensar_filtered/images/test"]
                    for candidate_dir in candidate_dirs:
                        if os.path.exists(candidate_dir):
                            sample_files = [f for f in os.listdir(candidate_dir) if f.endswith(".jpg")]
                            if sample_files:
                                import shutil
                                shutil.copy2(os.path.join(candidate_dir, sample_files[0]), raster_path)
                                break

            # If still missing, generate deterministic synthetic backscatter array (offline demo fallback only)
            if not os.path.exists(raster_path):
                if is_real_scene:
                    raise RuntimeError(
                        f"{mod_upper} raster data unavailable for real observation {scene_id}: "
                        "cannot execute vessel detection on synthetic data"
                    )
                elif mod_upper == "SLC":
                    raise RuntimeError(
                        f"SLC raster data unavailable for {scene_id}: cannot execute vessel detection on synthetic data"
                    )
                arr = np.random.normal(loc=40.0, scale=15.0, size=(640, 640, 3)).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(raster_path)

            scene_img = np.array(Image.open(raster_path).convert("RGB"))
            h, w = scene_img.shape[:2]

            # Use dedicated preprocessor for modality - never mix GRD and SLC
            if mod_upper == "SLC":
                tiles = self.slc_preprocessor.generate_tiles(scene_img, modality="SLC")
            else:
                tiles = self.preprocessor.generate_tiles(scene_img, modality="GRD")

            # 3. INFERENCE PHASE
            job.status = SARJobStatus.INFERENCE
            job.progress_percent = 60
            job.message = f"Executing YOLO26s {mod_upper} vessel detection and georeferencing"

            footprint_geom = scene_summary.footprint if scene_summary else {"type": "Polygon", "coordinates": [[[-0.5, 50.0], [1.5, 50.2], [1.3, 50.9], [-0.7, 50.7], [-0.5, 50.0]]]}
            corner_coords = SARGeoreferencer.extract_corners_from_footprint(footprint_geom)

            if scene_summary and scene_summary.scene_metadata:
                effective_metadata.update(scene_summary.scene_metadata)
            if burst_meta:
                effective_metadata.update(burst_meta)

            detector = YOLO26sVesselDetector.get_instance(modality=mod_upper)
            detections = detector.predict_scene_tiles(
                tiles=tiles,
                corner_coords=corner_coords,
                parent_width=w,
                parent_height=h,
                confidence_threshold=0.25,
                scene_metadata=effective_metadata,
            )

            # 4. AIS CORRELATION PHASE
            job.status = SARJobStatus.CORRELATING
            job.progress_percent = 80
            job.message = "Correlating detections against AIS within ±15 minutes and 3.0 km"

            acq_time = scene_summary.acquisition_time if scene_summary else datetime.now(UTC)
            detection_items: list[SARDetectionItem] = []
            matched_count = 0
            unmatched_count = 0
            coverage_unavailable_count = 0

            async with session_factory() as session:
                for det in detections:
                    lat = det["latitude"]
                    lon = det["longitude"]

                    # Run spatiotemporal correlation
                    corr_result = await self.correlation_engine.correlate_detection(
                        session=session,
                        lat=lat,
                        lon=lon,
                        acquisition_time=acq_time,
                    )

                    match_status = corr_result["match_status"]
                    if match_status == MatchStatus.MATCHED:
                        matched_count += 1
                    elif match_status == MatchStatus.UNMATCHED:
                        unmatched_count += 1
                    else:
                        coverage_unavailable_count += 1

                    # Persist SARDetection and AISSARCorrelation
                    geom_wkt = f"SRID=4326;POINT({lon} {lat})"
                    db_det = SARDetection(
                        detection_id=det["detection_id"],
                        scene_id=scene_id,
                        geom=geom_wkt,
                        latitude=lat,
                        longitude=lon,
                        pixel_bbox=det["pixel_bbox"],
                        confidence=det["confidence"],
                        model_version=det["model_version"],
                        length_m=det.get("length_m"),
                        heading_deg=det.get("heading_deg"),
                    )
                    session.add(db_det)
                    await session.flush()

                    db_corr = AISSARCorrelation(
                        sar_detection_id=db_det.id,
                        candidate_mmsi=corr_result["candidate_mmsi"],
                        time_difference_seconds=corr_result["time_difference_seconds"],
                        distance_km=corr_result["distance_km"],
                        match_status=match_status.value,
                        matching_method="spatiotemporal_nearest",
                        provenance=corr_result["provenance"],
                    )
                    session.add(db_corr)

                    det_provenance = {
                        **(corr_result.get("provenance") or {}),
                        "geolocation_method": "piecewise_gcp_grid" if ("geolocation_grid_points" in effective_metadata and effective_metadata["geolocation_grid_points"]) else "bilinear_footprint",
                        "modality": mod_upper,
                    }
                    if "source_tile" in det:
                        det_provenance["source_tile"] = det["source_tile"]
                    if "coord_transform" in det:
                        det_provenance["coord_transform"] = det["coord_transform"]
                    if det.get("duplicate_provenance"):
                        det_provenance["duplicate_provenance"] = det["duplicate_provenance"]
                    if "rangePixelSpacing" in effective_metadata:
                        det_provenance["range_pixel_spacing_m"] = effective_metadata["rangePixelSpacing"]
                        det_provenance["azimuth_pixel_spacing_m"] = effective_metadata.get("azimuthPixelSpacing")

                    # Validate geographic land/water context
                    geo_ctx = self.geographic_validator.validate_candidate(lat=lat, lon=lon)
                    det_provenance["geographic_context"] = geo_ctx.to_dict()

                    detection_items.append(
                        SARDetectionItem(
                            detection_id=det["detection_id"],
                            scene_id=scene_id,
                            acquisition_time=acq_time,
                            latitude=lat,
                            longitude=lon,
                            pixel_bbox=det["pixel_bbox"],
                            confidence=det["confidence"],
                            model_version=det["model_version"],
                            modality=mod_upper,
                            length_m=det.get("length_m"),
                            heading_deg=det.get("heading_deg"),
                            match_status=match_status,
                            surface_context=geo_ctx.status,
                            is_operational_vessel=geo_ctx.is_operational_vessel,
                            candidate_mmsi=corr_result["candidate_mmsi"],
                            time_difference_seconds=corr_result["time_difference_seconds"],
                            distance_km=corr_result["distance_km"],
                            provenance=det_provenance,
                        )
                    )

                await session.commit()

            # 5. COMPLETE PHASE
            operational_vessels = [d for d in detection_items if d.is_operational_vessel]
            clutter_candidates = [d for d in detection_items if not d.is_operational_vessel]

            summary_parts = [
                f"{matched_count} AIS-MATCHED",
                f"{unmatched_count} AIS-UNMATCHED",
            ]
            if coverage_unavailable_count > 0:
                summary_parts.append(f"{coverage_unavailable_count} AIS COVERAGE UNAVAILABLE")
            summary_str = ", ".join(summary_parts)

            if clutter_candidates:
                job.message = (
                    f"Analysis complete: {len(operational_vessels)} operational surface contacts detected "
                    f"({summary_str}, {len(clutter_candidates)} terrestrial clutter candidates flagged)"
                )
            else:
                job.message = f"Analysis complete: {len(detection_items)} vessels detected ({summary_str})"

            job.status = SARJobStatus.COMPLETE
            job.progress_percent = 100
            job.detections_count = len(operational_vessels) if mod_upper == "SLC" else len(detection_items)
            job.matched_count = matched_count
            job.unmatched_count = unmatched_count
            job.coverage_unavailable_count = coverage_unavailable_count
            job.completed_at = datetime.now(UTC)
            job.detections = detection_items

            logger.info("SAR job %s finished successfully: %d detections", job_id, len(detection_items))

        except Exception as exc:
            logger.exception("SAR workflow failed for job %s: %s", job_id, exc)
            job.status = SARJobStatus.FAILED
            job.progress_percent = 100
            job.message = f"SAR analysis failed: {exc!s}"
            job.error_detail = str(exc)

    async def get_scene_detections(
        self,
        scene_id: str,
        session: AsyncSession,
    ) -> list[SARDetectionItem]:
        """Query persisted detections for a scene with their AIS correlation join."""
        query = text("""
            SELECT
                d.detection_id,
                d.scene_id,
                s.acquisition_time,
                d.latitude,
                d.longitude,
                d.pixel_bbox,
                d.confidence,
                d.model_version,
                d.length_m,
                d.heading_deg,
                c.match_status,
                c.candidate_mmsi,
                c.time_difference_seconds,
                c.distance_km,
                c.provenance
            FROM sar_detection d
            JOIN sar_scene s ON d.scene_id = s.scene_id
            LEFT JOIN ais_sar_correlation c ON d.id = c.sar_detection_id
            WHERE d.scene_id = :scene_id
            ORDER BY d.confidence DESC;
        """)

        result = await session.execute(query, {"scene_id": scene_id})
        items: list[SARDetectionItem] = []

        for row in result.fetchall():
            (
                det_id,
                sc_id,
                acq_time,
                lat,
                lon,
                p_bbox,
                conf,
                m_ver,
                len_m,
                hdg_deg,
                m_status,
                c_mmsi,
                t_diff,
                dist_km,
                prov,
            ) = row

            status_enum = MatchStatus.MATCHED if m_status == "AIS-MATCHED" else MatchStatus.UNMATCHED

            items.append(
                SARDetectionItem(
                    detection_id=det_id,
                    scene_id=sc_id,
                    acquisition_time=acq_time,
                    latitude=float(lat),
                    longitude=float(lon),
                    pixel_bbox=p_bbox,
                    confidence=float(conf),
                    model_version=m_ver,
                    length_m=float(len_m) if len_m is not None else None,
                    heading_deg=float(hdg_deg) if hdg_deg is not None else None,
                    match_status=status_enum,
                    candidate_mmsi=int(c_mmsi) if c_mmsi is not None else None,
                    time_difference_seconds=float(t_diff) if t_diff is not None else None,
                    distance_km=float(dist_km) if dist_km is not None else None,
                    provenance=prov,
                )
            )

        return items
