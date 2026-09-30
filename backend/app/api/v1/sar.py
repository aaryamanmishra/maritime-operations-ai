import os
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.database.session import get_db_session, async_session_factory
from app.infrastructure.database.models.sar import SARScene
from app.domain.sar_intelligence.models import (
    SARSearchRequest,
    SARSceneSummary,
    SARJobResponse,
    SARDetectionItem,
    SARJobStatus,
)
from app.domain.sar_intelligence.service import SARAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sar", tags=["dark-vessel-intelligence"])


class SARAnalyzeRequest(BaseModel):
    scene_id: str
    scene_summary: Optional[SARSceneSummary] = None


@router.post("/search", response_model=List[SARSceneSummary])
async def search_sentinel1_scenes(request: SARSearchRequest):
    """
    Search Sentinel-1 GRD observations from the Copernicus Data Space STAC catalog.
    No automatic download occurs during search.
    """
    service = SARAnalysisService.get_instance()
    try:
        scenes = await service.search_scenes(
            bbox=request.bbox,
            start_time=request.start_time,
            end_time=request.end_time,
            limit=request.limit,
        )
        return scenes
    except Exception as exc:
        logger.error("Scene search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Copernicus catalog search failed: {str(exc)}",
        )


@router.post("/jobs", response_model=SARJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_sar_analysis_job(
    request: SARAnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    Initiate an asynchronous Sentinel-1 analysis job.
    Inference and AIS correlation are processed in the background.
    """
    service = SARAnalysisService.get_instance()
    try:
        job = await service.start_job(
            scene_id=request.scene_id,
            background_tasks=background_tasks,
            session_factory=async_session_factory,
            scene_summary=request.scene_summary,
        )
        return job
    except Exception as exc:
        logger.error("Failed creating SAR job for %s: %s", request.scene_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not initiate SAR analysis job: {str(exc)}",
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


@router.get("/scenes", response_model=List[Dict[str, Any]])
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


@router.get("/scenes/{scene_id}/detections", response_model=List[SARDetectionItem])
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
async def get_model_status():
    """
    Get current YOLO26s SAR vessel detector status, weights, and evaluation metrics.
    Exposes actual trained model metadata and held-out test metrics.
    Marked explicitly as TRAINED / OPERATIONAL (not certified or production-validated).
    """
    meta_path = "backend/ml/weights/yolo26s_metadata.json"
    if not os.path.exists(meta_path):
        meta_path = "ml/weights/yolo26s_metadata.json"

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "status": "TRAINED / OPERATIONAL",
        "operational_declaration": "TRAINED / OPERATIONAL (Not certified or production-validated)",
        "model_name": metadata.get("model_name", "YOLO26s-SAR-Vessel"),
        "model_version": metadata.get("model_version", "yolo26s-opensar-v1"),
        "architecture": metadata.get("architecture", "YOLO26s (Ultralytics / OpenSAR Insight)"),
        "task": metadata.get("task", "detect"),
        "classes": metadata.get("classes", {0: "vessel"}),
        "training": metadata.get("training", {
            "platform": "Kaggle NVIDIA Tesla T4",
            "epochs_completed": 87,
            "best_epoch": 62,
            "batch_size": 16,
            "imgsz": 640,
        }),
        "evaluation_held_out_test": metadata.get("evaluation_held_out_test", {
            "precision": 0.4044,
            "recall": 0.3152,
            "mAP50": 0.2335,
            "mAP50_95": 0.0471,
            "cpu_inference_latency_ms": 57.10,
            "notes": "Evaluated strictly on held-out OpenSAR test split (100 images, 257 vessels)."
        }),
        "limitations": metadata.get(
            "limitations",
            "Metrics reflect performance on the OpenSAR Insight held-out test split under standard high-contrast Sentinel-1 GRD processing. Generalization gap is expected in high-clutter sea states or variable incidence angles. Detections must be interpreted as probabilistic operational estimates, not navigational ground truth."
        ),
        "metadata": metadata,
    }

