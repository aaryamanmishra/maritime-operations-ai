from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


class GeoCoordinate(BaseModel):
    """Geographic coordinate in WGS84."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees [-90, 90]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees [-180, 180]")


class VesselCharacteristics(BaseModel):
    """
    Validated vessel physical and performance parameters.
    No parameters are silently invented; defensible defaults are documented where optional.
    """
    vessel_type: Literal["cargo", "tanker", "passenger", "fishing", "tug", "service", "other"] = Field(
        ..., description="Standard vessel category"
    )
    length_m: float = Field(..., gt=5.0, lt=500.0, description="Overall length in meters")
    beam_m: float = Field(..., gt=1.0, lt=100.0, description="Beam (breadth) in meters")
    draft_m: float = Field(..., gt=0.5, lt=35.0, description="Design draft in meters")
    design_speed_kn: float = Field(..., gt=2.0, lt=45.0, description="Calm-water design speed in knots")
    
    # Optional parameters with engineering defaults
    displacement_t: Optional[float] = Field(None, gt=10.0, description="Loaded displacement in metric tonnes")
    propulsion_power_kw: Optional[float] = Field(None, gt=50.0, description="Installed Maximum Continuous Rating (MCR) in kW")
    engine_efficiency: float = Field(0.42, gt=0.2, lt=0.6, description="Brake thermal efficiency of engine (default 0.42)")
    propulsive_efficiency: float = Field(0.65, gt=0.3, lt=0.85, description="Quasi-propulsive coefficient eta_D (default 0.65)")

    @field_validator("beam_m")
    @classmethod
    def validate_beam(cls, v: float, info) -> float:
        length = info.data.get("length_m")
        if length and v >= length:
            raise ValueError("Beam cannot exceed or equal vessel length")
        return v


class FuelParameters(BaseModel):
    """Fuel characteristics and unit economic cost."""
    fuel_price_per_tonne: float = Field(..., gt=50.0, lt=3000.0, description="Fuel market price in USD per metric tonne")
    fuel_type: Literal["VLSFO", "MGO", "HFO", "LNG", "METHANOL"] = Field(
        "VLSFO", description="Bunker fuel type"
    )


class RouteOptimizationRequest(BaseModel):
    """Validated input specification for AI Weather-Aware Routing."""
    origin: GeoCoordinate
    destination: GeoCoordinate
    departure_time: datetime = Field(..., description="Departure timestamp in UTC")
    vessel: VesselCharacteristics
    fuel: FuelParameters
    optimization_priority: Literal["balanced", "fuel_min", "time_min"] = Field(
        "balanced", description="Optimization objective weighting"
    )

    @field_validator("origin", "destination")
    @classmethod
    def validate_coords(cls, v: GeoCoordinate) -> GeoCoordinate:
        return v


class MarineWeatherConditions(BaseModel):
    """Hourly marine meteorological and oceanographic conditions at a route waypoint."""
    timestamp: datetime
    wave_height_m: float = Field(..., ge=0.0, description="Significant wave height (Hs) in meters")
    wave_direction_deg: float = Field(..., ge=0.0, le=360.0, description="Wave direction in degrees from north")
    wave_period_s: float = Field(..., ge=0.0, description="Peak wave period in seconds")
    wind_speed_mps: float = Field(..., ge=0.0, description="10m wind speed in meters per second")
    wind_direction_deg: float = Field(..., ge=0.0, le=360.0, description="Wind direction in degrees from north")
    ocean_current_velocity_mps: float = Field(0.0, ge=0.0, description="Surface current speed in m/s")
    ocean_current_direction_deg: float = Field(0.0, ge=0.0, le=360.0, description="Surface current direction in degrees")
    sea_surface_temperature_c: float = Field(15.0, description="Sea surface temperature in Celsius")
    swell_wave_height_m: float = Field(0.0, ge=0.0, description="Swell wave height in meters")
    relative_wind_direction_deg: Optional[float] = None
    relative_wave_direction_deg: Optional[float] = None


class RouteSegment(BaseModel):
    """Individual navigational leg with full environmental provenance and estimated performance."""
    segment_index: int
    start_point: GeoCoordinate
    end_point: GeoCoordinate
    distance_nm: float
    bearing_deg: float
    estimated_departure: datetime
    estimated_arrival: datetime
    duration_hours: float
    
    # Environmental conditions along this segment
    weather: MarineWeatherConditions

    # AI Model & Physics outputs (explicitly labeled as estimates)
    calm_reference_speed_kn: float
    predicted_speed_kn: float
    speed_loss_kn: float
    current_speed_contribution_kn: float
    total_resistance_kn: float
    estimated_brake_power_kw: float
    estimated_energy_kwh: float
    estimated_fuel_tonnes: float
    estimated_fuel_cost_usd: float


class RouteSummary(BaseModel):
    """Aggregate metrics and exposure statistics for a route."""
    total_distance_nm: float
    estimated_duration_hours: float
    departure_time: datetime
    estimated_eta: datetime
    mean_speed_kn: float
    mean_speed_loss_kn: float
    total_estimated_energy_kwh: float
    total_estimated_fuel_tonnes: float
    total_estimated_cost_usd: float
    max_wave_height_m: float
    max_wind_speed_mps: float
    avg_wave_height_m: float
    weather_exposure_level: Literal["calm", "moderate", "rough", "severe"]


class NavigationalRoute(BaseModel):
    """Complete route result with GeoJSON LineString geometry and segment details."""
    route_type: Literal["base_maritime", "weather_aware"]
    geometry: Dict[str, Any]  # GeoJSON LineString
    summary: RouteSummary
    segments: List[RouteSegment]


class RouteOptimizationResponse(BaseModel):
    """Production response delivering both Base Maritime and AI Weather-Aware routes."""
    request: RouteOptimizationRequest
    base_route: NavigationalRoute
    weather_aware_route: NavigationalRoute
    route_diverged: bool
    summary_comparison: Dict[str, Any]
    model_version: str = "TFT-v1.0.0-NOAA-TrackA"
    weather_provider: str = "Open-Meteo Marine API (CC-BY 4.0)"
    disclaimer: str = (
        "OPERATIONAL NOTICE: All fuel, energy, duration, and performance values are computational "
        "ESTIMATES based on the MARIS-Forecast TFT surrogate and standard hydrodynamic models. "
        "They do NOT constitute certified navigational advice, guaranteed-safe routing, or certified fuel savings."
    )
