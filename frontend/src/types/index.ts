export interface HealthLiveResponse {
  status: string;
  timestamp: string;
}

export interface HealthReadyResponse {
  status: string;
  database: string;
  redis: string;
  timestamp: string;
}

export type ActiveModule = 'traffic' | 'routing' | 'dark-vessel';

export interface Vessel {
  mmsi: number;
  name: string | null;
  imo: number | null;
  callsign: string | null;
  ship_type: number | null;
  destination: string | null;
  length: number | null;
  beam: number | null;
  draught: number | null;
  latitude: number;
  longitude: number;
  sog: number | null;
  cog: number | null;
  heading: number | null;
  nav_status: number | null;
  timestamp: string;
  updated_at: string;
}

export interface TrackPoint {
  latitude: number;
  longitude: number;
  timestamp: string;
  sog: number | null;
  cog: number | null;
  heading: number | null;
}

export interface VesselDetails extends Vessel {
  recent_track: TrackPoint[];
}

export interface PipelineStatus {
  connection_state: 'connected' | 'connecting' | 'reconnecting' | 'disconnected';
  messages_received: number;
  normalized_count: number;
  stream_writes: number;
  persisted_count: number;
  last_message_at: string | null;
  connected_at: string | null;
  total_vessels_in_db: number;
  bbox: [number, number, number, number];
  updated_at?: string;
}

export type IngestionStatus = 'LIVE' | 'STALE' | 'DISCONNECTED';
