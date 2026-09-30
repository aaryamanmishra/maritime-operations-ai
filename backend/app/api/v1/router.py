from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.vessels import router as vessels_router
from app.api.v1.routing import router as routing_router
from app.api.v1.sar import router as sar_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(vessels_router)
api_v1_router.include_router(routing_router)
api_v1_router.include_router(sar_router)

# Also expose under /api for backward compatibility with Phase 1 probes
api_legacy_router = APIRouter(prefix="/api")
api_legacy_router.include_router(health_router)
api_legacy_router.include_router(vessels_router)
api_legacy_router.include_router(routing_router)
api_legacy_router.include_router(sar_router)
