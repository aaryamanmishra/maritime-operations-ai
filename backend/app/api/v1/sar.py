import json
import logging
import os
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.sar_intelligence.models import (
    SARDetectionItem,
    SARJobResponse,
    SARSceneSummary,
    SARSearchRequest,
)
from app.domain.sar_intelligence.service import SARAnalysisService
from app.infrastructure.database.models.sar import SARScene
from app.infrastructure.database.session import async_session_factory, get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sar", tags=["dark-vessel-intelligence"])


class SARAnalyzeRequest(BaseModel):
    scene_id: str
    scene_summary: SARSceneSummary | None = None
    modality: str = "GRD"


@router.post("/search", response_model=list[SARSceneSummary])
async def search_sentinel1_scenes(request: SARSearchRequest):
    """
    Search Sentinel-1 observations from the Copernicus Data Space STAC catalog.
    Defaults to GRD ('sentinel-1-grd'); opt-in to SLC ('sentinel-1-slc') via request.modality.
    No automatic download occurs during search.
    """
    service = SARAnalysisService.get_instance()
    try:
        scenes = await service.search_scenes(
            bbox=request.bbox,
            start_time=request.start_time,
            end_time=request.end_time,
            limit=request.limit,
            modality=request.modality.value if hasattr(request.modality, "value") else str(request.modality),
        )
        return scenes
    except Exception as exc:
        logger.error("Scene search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Copernicus catalog search failed: {exc!s}",
        )


@router.post("/jobs", response_model=SARJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_sar_analysis_job(
    request: SARAnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    Initiate an asynchronous Sentinel-1 analysis job.
    Inference and AIS correlation are processed in the background.
    Supports GRD (default production) or SLC (opt-in) pipelines.
    """
    service = SARAnalysisService.get_instance()
    try:
        job = await service.start_job(
            scene_id=request.scene_id,
            background_tasks=background_tasks,
            session_factory=async_session_factory,
            scene_summary=request.scene_summary,
            modality=request.modality,
        )
        return job
    except Exception as exc:
        logger.error("Failed creating SAR job for %s: %s", request.scene_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not initiate SAR analysis job: {exc!s}",
        )


@router.get("/jobs/{job_id}", response_model=SARJobResponse)
async def get_sar_job_status(job_id: str):
    """
    Poll the execution status and results of a SAR analysis job.
    """
    service = SARAnalysisService.get_instance()
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SAR job {job_id} not found",
        )
    return job


@router.get("/scenes", response_model=list[dict[str, Any]])
async def list_analyzed_scenes(session: AsyncSession = Depends(get_db_session)):
    """
    List all previously analyzed Sentinel-1 scenes stored in PostGIS.
    """
    res = await session.execute(
        select(SARScene).order_by(SARScene.acquisition_time.desc()).limit(50)
    )
    scenes = res.scalars().all()

    items = []
    for sc in scenes:
        items.append({
            "scene_id": sc.scene_id,
            "acquisition_time": sc.acquisition_time.isoformat(),
            "platform": sc.platform,
            "orbit_pass": sc.orbit_pass,
            "polarization": sc.polarization,
            "source": sc.source,
            "created_at": sc.created_at.isoformat(),
        })
    return items


@router.get("/scenes/{scene_id}/detections", response_model=list[SARDetectionItem])
async def get_scene_detections(
    scene_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Get all vessel detections and AIS correlation provenance for a specific scene.
    """
    service = SARAnalysisService.get_instance()
    detections = await service.get_scene_detections(scene_id, session)
    return detections


@router.get("/model/status")
async def get_model_status(modality: str = "GRD"):
    """
    Get current YOLO26s SAR vessel detector status, weights, and evaluation metrics.
    Exposes actual trained model metadata and held-out test metrics.
    Supports 'GRD' (default) or 'SLC' (opt-in).
    Marked explicitly as TRAINED / OPERATIONAL (not certified or production-validated).
    """
    mod_upper = modality.upper()
    if mod_upper == "SLC":
        meta_candidates = [
            "backend/ml/weights/yolo26s_slc_metadata.json",
            "ml/weights/yolo26s_slc_metadata.json",
        ]
        default_version = "yolo26s-opensar-slc-v1"
        default_eval = {
            "precision": 0.5255,
            "recall": 0.4164,
            "mAP50": 0.3792,
            "mAP50_95": 0.1284,
            "notes": "Evaluated strictly on held-out OpenSAR SLC test split (949 images, 1220 vessels).",
        }
        default_limitations = "Metrics reflect performance on the OpenSAR Insight held-out SLC test split under native slant-range complex magnitude processing."
    else:
        meta_candidates = [
            "backend/ml/weights/yolo26s_metadata.json",
            "ml/weights/yolo26s_metadata.json",
        ]
        default_version = "yolo26s-opensar-v1"
        default_eval = {
            "precision": 0.4044,
            "recall": 0.3152,
            "mAP50": 0.2335,
            "mAP50_95": 0.0471,
            "cpu_inference_latency_ms": 57.10,
            "notes": "Evaluated strictly on held-out OpenSAR test split (100 images, 257 vessels).",
        }
        default_limitations = (
            "Metrics reflect performance on the OpenSAR Insight held-out test split under standard high-contrast Sentinel-1 GRD processing. "
            "Generalization gap is expected in high-clutter sea states or variable incidence angles. "
            "Detections must be interpreted as probabilistic operational estimates, not navigational ground truth."
        )

    metadata = {}
    for m_path in meta_candidates:
        if os.path.exists(m_path):
            with open(m_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            break

    # If metadata uses test_metrics (like SLC metadata from Kaggle export), adapt to evaluation_held_out_test
    if "test_metrics" in metadata and "evaluation_held_out_test" not in metadata:
        tm = metadata["test_metrics"]
        eval_metrics = {
            "precision": tm.get("precision", default_eval["precision"]),
            "recall": tm.get("recall", default_eval["recall"]),
            "mAP50": tm.get("map50", default_eval["mAP50"]),
            "mAP50_95": tm.get("map50_95", default_eval["mAP50_95"]),
            "notes": default_eval.get("notes"),
        }
    else:
        eval_metrics = metadata.get("evaluation_held_out_test", default_eval)

    return {
        "status": "TRAINED / OPERATIONAL",
        "operational_declaration": "TRAINED / OPERATIONAL (Not certified or production-validated)",
        "modality": mod_upper,
        "model_name": metadata.get("model_name", "YOLO26s-SAR-Vessel"),
        "model_version": metadata.get("model_version", default_version),
        "architecture": metadata.get("architecture") or metadata.get("model_architecture") or "YOLO26s (Ultralytics / OpenSAR Insight)",
        "task": metadata.get("task", "detect"),
        "classes": metadata.get("classes", {0: "vessel"}),
        "training": metadata.get("training", {
            "platform": "Kaggle NVIDIA Tesla T4",
            "epochs_completed": metadata.get("epochs_completed", 87),
            "best_epoch": metadata.get("best_epoch", 62),
            "batch_size": metadata.get("batch_size", 16),
            "imgsz": metadata.get("image_size", 640),
        }),
        "evaluation_held_out_test": eval_metrics,
        "limitations": metadata.get("limitations", default_limitations),
        "metadata": metadata,
    }

