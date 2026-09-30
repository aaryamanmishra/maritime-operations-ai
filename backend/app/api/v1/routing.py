from fastapi import APIRouter, HTTPException, status
from app.core.logging import logger
from app.domain.weather_routing.models import (
    RouteOptimizationRequest,
    RouteOptimizationResponse,
)
from app.routing.optimizer import WeatherAwareRouteOptimizer

router = APIRouter(prefix="/routing", tags=["Weather-Aware Routing"])

optimizer = WeatherAwareRouteOptimizer()


@router.post(
    "/optimize",
    response_model=RouteOptimizationResponse,
    summary="Compute AI Weather-Aware Maritime Route",
    description=(
        "Optimizes a voyage route between origin and destination using the SeaRoute maritime network, "
        "Open-Meteo marine weather conditions, and a Temporal Fusion Transformer (TFT) surrogate model. "
        "Returns both the Base Maritime Route and the optimized Weather-Aware Route with detailed segment provenance."
    ),
)
async def optimize_maritime_route(request: RouteOptimizationRequest) -> RouteOptimizationResponse:
    try:
        response = await optimizer.optimize_route(request)
        return response
    except ValueError as ve:
        logger.warning(f"Routing validation error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except RuntimeError as re:
        logger.error(f"Routing runtime error: {re}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream routing or weather service failure: {re}",
        )
    except Exception as exc:
        logger.error(f"Unexpected routing optimization exception: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during route optimization.",
        )


@router.get(
    "/defaults",
    summary="Get standard vessel defaults",
    description="Returns standard defensible design parameters by vessel category for operational prefill.",
)
async def get_vessel_defaults():
    return {
        "cargo": {
            "vessel_type": "cargo",
            "length_m": 190.0,
            "beam_m": 28.0,
            "draft_m": 10.5,
            "design_speed_kn": 14.5,
            "fuel_price_per_tonne": 620.0,
            "fuel_type": "VLSFO",
        },
        "tanker": {
            "vessel_type": "tanker",
            "length_m": 245.0,
            "beam_m": 42.0,
            "draft_m": 14.0,
            "design_speed_kn": 13.5,
            "fuel_price_per_tonne": 620.0,
            "fuel_type": "VLSFO",
        },
        "passenger": {
            "vessel_type": "passenger",
            "length_m": 180.0,
            "beam_m": 26.0,
            "draft_m": 6.8,
            "design_speed_kn": 19.0,
            "fuel_price_per_tonne": 780.0,
            "fuel_type": "MGO",
        },
        "fishing": {
            "vessel_type": "fishing",
            "length_m": 45.0,
            "beam_m": 9.5,
            "draft_m": 4.2,
            "design_speed_kn": 10.0,
            "fuel_price_per_tonne": 780.0,
            "fuel_type": "MGO",
        },
        "tug": {
            "vessel_type": "tug",
            "length_m": 32.0,
            "beam_m": 11.0,
            "draft_m": 5.2,
            "design_speed_kn": 11.5,
            "fuel_price_per_tonne": 780.0,
            "fuel_type": "MGO",
        },
    }
