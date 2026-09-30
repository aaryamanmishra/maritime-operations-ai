export interface GeoCoordinate {
  latitude: number;
  longitude: number;
}

export interface VesselCharacteristics {
  vessel_type: "cargo" | "tanker" | "passenger" | "fishing" | "tug" | "service" | "other";
  length_m: number;
  beam_m: number;
  draft_m: number;
  design_speed_kn: number;
  displacement_t?: number;
  propulsion_power_kw?: number;
  engine_efficiency?: number;
  propulsive_efficiency?: number;
}

export interface FuelParameters {
  fuel_price_per_tonne: number;
  fuel_type: "VLSFO" | "MGO" | "HFO" | "LNG" | "METHANOL";
}

export interface RouteOptimizationRequest {
  origin: GeoCoordinate;
  destination: GeoCoordinate;
  departure_time: string;
  vessel: VesselCharacteristics;
  fuel: FuelParameters;
  optimization_priority: "balanced" | "fuel_min" | "time_min";
}

export interface MarineWeatherConditions {
  timestamp: string;
  wave_height_m: number;
  wave_direction_deg: number;
  wave_period_s: number;
  wind_speed_mps: number;
  wind_direction_deg: number;
  ocean_current_velocity_mps: number;
  ocean_current_direction_deg: number;
  sea_surface_temperature_c: number;
  swell_wave_height_m: number;
}

export interface RouteSegment {
  segment_index: number;
  start_point: GeoCoordinate;
  end_point: GeoCoordinate;
  distance_nm: number;
  bearing_deg: number;
  estimated_departure: string;
  estimated_arrival: string;
  duration_hours: number;
  weather: MarineWeatherConditions;
  calm_reference_speed_kn: number;
  predicted_speed_kn: number;
  speed_loss_kn: number;
  current_speed_contribution_kn: number;
  total_resistance_kn: number;
  estimated_brake_power_kw: number;
  estimated_energy_kwh: number;
  estimated_fuel_tonnes: number;
  estimated_fuel_cost_usd: number;
}

export interface RouteSummary {
  total_distance_nm: number;
  estimated_duration_hours: number;
  departure_time: string;
  estimated_eta: string;
  mean_speed_kn: number;
  mean_speed_loss_kn: number;
  total_estimated_energy_kwh: number;
  total_estimated_fuel_tonnes: number;
  total_estimated_cost_usd: number;
  max_wave_height_m: number;
  max_wind_speed_mps: number;
  avg_wave_height_m: number;
  weather_exposure_level: "calm" | "moderate" | "rough" | "severe";
}

export interface NavigationalRoute {
  route_type: "base_maritime" | "weather_aware";
  geometry: {
    type: string;
    coordinates: number[][];
  };
  summary: RouteSummary;
  segments: RouteSegment[];
}

export interface RouteOptimizationResponse {
  request: RouteOptimizationRequest;
  base_route: NavigationalRoute;
  weather_aware_route: NavigationalRoute;
  route_diverged: boolean;
  summary_comparison: {
    distance_difference_nm: number;
    duration_difference_hours: number;
    estimated_fuel_difference_tonnes: number;
    estimated_cost_difference_usd: number;
    base_max_wave_height_m: number;
    optimized_max_wave_height_m: number;
    wave_exposure_reduction_m: number;
  };
  model_version: string;
  weather_provider: string;
  disclaimer: string;
}

const API_BASE = "/api/v1";

export async function fetchVesselDefaults(): Promise<Record<string, VesselCharacteristics & FuelParameters>> {
  const resp = await fetch(`${API_BASE}/routing/defaults`);
  if (!resp.ok) {
    throw new Error(`Failed to load vessel defaults: ${resp.statusText}`);
  }
  return resp.json();
}

export async function optimizeRoute(req: RouteOptimizationRequest): Promise<RouteOptimizationResponse> {
  const resp = await fetch(`${API_BASE}/routing/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `Routing error: ${resp.status}`);
  }
  return resp.json();
}
