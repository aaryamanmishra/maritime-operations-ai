export interface SARSearchRequest {
  bbox: [number, number, number, number];
  start_time: string;
  end_time: string;
  limit?: number;
}

export interface SARSceneSummary {
  scene_id: string;
  acquisition_time: string;
  platform: string;
  orbit_pass?: string;
  orbit_number?: number;
  footprint: {
    type: string;
    coordinates: number[][][];
  };
  polarization: string;
  processing_level: string;
  source: string;
  quicklook_url?: string;
  download_url?: string;
  scene_metadata?: Record<string, any>;
}

export type MatchStatus = "AIS-MATCHED" | "AIS-UNMATCHED";

export interface SARDetectionItem {
  detection_id: string;
  scene_id: string;
  acquisition_time: string;
  latitude: number;
  longitude: number;
  pixel_bbox?: {
    ymin: number;
    xmin: number;
    ymax: number;
    xmax: number;
  };
  confidence: number;
  model_version: string;
  length_m?: number;
  heading_deg?: number;
  match_status: MatchStatus;
  candidate_mmsi?: number;
  time_difference_seconds?: number;
  distance_km?: number;
  matching_method?: string;
  provenance?: {
    explanation?: string;
    matching_thresholds?: Record<string, any>;
    sar_acquisition_time?: string;
    ais_observation_time?: string;
    distance_km?: number;
    time_difference_seconds?: number;
    algorithm_version?: string;
  };
}

export type SARJobStatus =
  | "QUEUED"
  | "DOWNLOADING"
  | "PREPROCESSING"
  | "INFERENCE"
  | "CORRELATING"
  | "COMPLETE"
  | "FAILED";

export interface SARJobResponse {
  job_id: string;
  scene_id: string;
  status: SARJobStatus;
  progress_percent: number;
  message: string;
  detections_count: number;
  matched_count: number;
  unmatched_count: number;
  created_at: string;
  completed_at?: string;
  detections: SARDetectionItem[];
  error_detail?: string;
}

export async function searchSARScenes(req: SARSearchRequest): Promise<SARSceneSummary[]> {
  const res = await fetch("/api/v1/sar/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Catalog search failed" }));
    throw new Error(err.detail || `Catalog search failed with status ${res.status}`);
  }
  return res.json();
}

export async function startSARJob(
  sceneId: string,
  sceneSummary?: SARSceneSummary
): Promise<SARJobResponse> {
  const res = await fetch("/api/v1/sar/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: sceneId, scene_summary: sceneSummary }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to initiate SAR job" }));
    throw new Error(err.detail || `Failed initiating SAR job with status ${res.status}`);
  }
  return res.json();
}

export async function fetchSARJobStatus(jobId: string): Promise<SARJobResponse> {
  const res = await fetch(`/api/v1/sar/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(`Job status check failed with status ${res.status}`);
  }
  return res.json();
}

export async function fetchAnalyzedScenes(): Promise<any[]> {
  const res = await fetch("/api/v1/sar/scenes");
  if (!res.ok) {
    throw new Error(`Failed to load analyzed scenes: ${res.status}`);
  }
  return res.json();
}

export async function fetchSceneDetections(sceneId: string): Promise<SARDetectionItem[]> {
  const res = await fetch(`/api/v1/sar/scenes/${sceneId}/detections`);
  if (!res.ok) {
    throw new Error(`Failed to fetch detections for scene ${sceneId}: ${res.status}`);
  }
  return res.json();
}
