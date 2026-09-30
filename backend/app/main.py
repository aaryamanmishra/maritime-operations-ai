from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_legacy_router, api_v1_router
from app.api.v1.ws import ws_router
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.core.middleware import CorrelationIdMiddleware
from app.infrastructure.database.session import engine
from app.infrastructure.redis.client import close_redis_connection, get_redis_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.PROJECT_NAME} in [{settings.ENVIRONMENT}] mode")
    get_redis_client()
    yield
    # Shutdown
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")
    await close_redis_connection()
    await engine.dispose()
    logger.info("Shutdown complete.")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        description="Maritime Operations AI - Foundation Service",
        lifespan=lifespan,
    )

    # Middlewares
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(api_v1_router)
    app.include_router(api_legacy_router)
    app.include_router(ws_router)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "project": settings.PROJECT_NAME,
            "version": "0.1.0",
            "status": "operational",
            "docs_url": "/docs",
        }

    return app


app = create_application()
