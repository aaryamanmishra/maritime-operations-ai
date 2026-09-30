import { HealthLiveResponse, HealthReadyResponse, Vessel, VesselDetails, PipelineStatus } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export async function fetchHealthLive(): Promise<HealthLiveResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health/live`);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function fetchHealthReady(): Promise<HealthReadyResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health/ready`);
  if (!response.ok && response.status !== 503) {
    throw new Error(`Readiness check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function fetchVesselsInBbox(
  minLat: number,
  minLon: number,
  maxLat: number,
  maxLon: number,
  limit: number = 500
): Promise<Vessel[]> {
  const params = new URLSearchParams({
    min_lat: minLat.toString(),
    min_lon: minLon.toString(),
    max_lat: maxLat.toString(),
    max_lon: maxLon.toString(),
    limit: limit.toString(),
  });

  const response = await fetch(`${API_BASE_URL}/api/v1/vessels?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch vessels: ${response.status}`);
  }
  return response.json();
}

export async function fetchVesselDetails(mmsi: number): Promise<VesselDetails> {
  const response = await fetch(`${API_BASE_URL}/api/v1/vessels/${mmsi}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch vessel details: ${response.status}`);
  }
  return response.json();
}

export async function fetchPipelineStatus(): Promise<PipelineStatus> {
  const response = await fetch(`${API_BASE_URL}/api/v1/vessels/pipeline/status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch pipeline status: ${response.status}`);
  }
  return response.json();
}
