import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Vessel, VesselDetails, PipelineStatus, IngestionStatus } from '../types';
import { fetchVesselsInBbox, fetchVesselDetails, fetchPipelineStatus } from '../services/api';
import { VesselDetailPanel } from './VesselDetailPanel';

import { RouteOptimizationResponse, GeoCoordinate } from '../services/routingApi';
import { SARSceneSummary, SARDetectionItem } from '../services/sarApi';

interface MaritimeMapProps {
  onStatusChange?: (status: IngestionStatus) => void;
  routeResponse?: RouteOptimizationResponse | null;
  origin?: GeoCoordinate | null;
  destination?: GeoCoordinate | null;
  activePickingMode?: "origin" | "destination" | null;
  onMapClickCoord?: (coord: GeoCoordinate) => void;
  selectedScene?: SARSceneSummary | null;
  sarDetections?: SARDetectionItem[];
  selectedDetection?: SARDetectionItem | null;
  onSelectDetection?: (detection: SARDetectionItem | null) => void;
}

const DEFAULT_CENTER: [number, number] = [0.25, 50.75]; // English Channel / Dover Strait
const DEFAULT_ZOOM = 7.5;
// OpenFreeMap "dark" style — free, no API key, CORS open, ODbL attribution required.
// Source: https://openfreemap.org  License: https://openfreemap.org/about/#license
const STYLE_URL = 'https://tiles.openfreemap.org/styles/dark';

export const MaritimeMap: React.FC<MaritimeMapProps> = ({
  onStatusChange,
  routeResponse,
  origin,
  destination,
  activePickingMode,
  onMapClickCoord,
  selectedScene,
  sarDetections = [],
  selectedDetection,
  onSelectDetection,
}) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const activePickingModeRef = useRef(activePickingMode);
  activePickingModeRef.current = activePickingMode;

  const onMapClickCoordRef = useRef(onMapClickCoord);
  onMapClickCoordRef.current = onMapClickCoord;

  const sarDetectionsRef = useRef(sarDetections);
  sarDetectionsRef.current = sarDetections;

  const onSelectDetectionRef = useRef(onSelectDetection);
  onSelectDetectionRef.current = onSelectDetection;

  const [vessels, setVessels] = useState<Map<number, Vessel>>(new Map());
  const [selectedMmsi, setSelectedMmsi] = useState<number | null>(null);
  const [selectedVessel, setSelectedVessel] = useState<VesselDetails | null>(null);
  const [loadingDetails, setLoadingDetails] = useState<boolean>(false);

  const [pipelineMetrics, setPipelineMetrics] = useState<PipelineStatus | null>(null);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [initialLoading, setInitialLoading] = useState<boolean>(true);

  // Compute honest ingestion status based on backend state & recent messages
  const computeIngestionStatus = useCallback((): IngestionStatus => {
    if (!wsConnected || !pipelineMetrics) {
      return 'DISCONNECTED';
    }
    if (pipelineMetrics.connection_state !== 'connected') {
      return 'DISCONNECTED';
    }
    if (!pipelineMetrics.last_message_at) {
      return 'STALE';
    }
    const lastMsgTime = new Date(pipelineMetrics.last_message_at).getTime();
    const now = Date.now();
    // If no message in the last 60 seconds, flag as stale
    if (now - lastMsgTime > 60000) {
      return 'STALE';
    }
    return 'LIVE';
  }, [wsConnected, pipelineMetrics]);

  const ingestionStatus = computeIngestionStatus();

  useEffect(() => {
    onStatusChange?.(ingestionStatus);
  }, [ingestionStatus, onStatusChange]);

  // Sync pipeline diagnostics every 5 seconds
  useEffect(() => {
    const syncStatus = async () => {
      try {
        const metrics = await fetchPipelineStatus();
        setPipelineMetrics(metrics);
      } catch (err) {
        // Backend unavailable
      }
    };
    syncStatus();
    const interval = setInterval(syncStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Update GeoJSON layers when vessels change
  const updateMapLayers = useCallback((vesselMap: Map<number, Vessel>) => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const source = map.getSource('vessels-source') as maplibregl.GeoJSONSource | undefined;
    if (!source) return;

    const features: any[] = [];
    vesselMap.forEach((v) => {
      features.push({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [v.longitude, v.latitude],
        },
        properties: {
          mmsi: v.mmsi,
          name: v.name || 'Unknown',
          sog: v.sog ?? 0,
          heading: v.heading ?? v.cog ?? 0,
          ship_type: v.ship_type ?? 0,
          isSelected: v.mmsi === selectedMmsi,
        },
      });
    });

    source.setData({
      type: 'FeatureCollection',
      features,
    });
  }, [selectedMmsi]);

  // Update track layer for selected vessel
  const updateTrackLayer = useCallback((trackPoints: { longitude: number; latitude: number }[]) => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const trackSource = map.getSource('track-source') as maplibregl.GeoJSONSource | undefined;
    if (!trackSource) return;

    if (!trackPoints || trackPoints.length < 2) {
      trackSource.setData({
        type: 'FeatureCollection',
        features: [],
      });
      return;
    }

    trackSource.setData({
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: trackPoints.map((p) => [p.longitude, p.latitude]),
      },
      properties: {},
    });
  }, []);

  // Fetch vessels for current viewport
  const fetchViewportVessels = useCallback(async () => {
    const map = mapRef.current;
    if (!map) return;

    const bounds = map.getBounds();
    const minLon = bounds.getWest();
    const minLat = bounds.getSouth();
    const maxLon = bounds.getEast();
    const maxLat = bounds.getNorth();

    // Notify backend WebSocket of new viewport bounding box
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'viewport',
        bbox: [minLon, minLat, maxLon, maxLat],
      }));
    }

    try {
      const data = await fetchVesselsInBbox(minLat, minLon, maxLat, maxLon, 500);
      setVessels((prev) => {
        const next = new Map(prev);
        data.forEach((v) => next.set(v.mmsi, v));
        updateMapLayers(next);
        return next;
      });
    } catch (err) {
      console.warn('Error fetching viewport vessels:', err);
    } finally {
      setInitialLoading(false);
    }
  }, [updateMapLayers]);

  // Connect WebSocket to /ws/vessels
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/vessels`;

    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        // Send initial viewport once connected
        const map = mapRef.current;
        if (map) {
          const bounds = map.getBounds();
          ws?.send(JSON.stringify({
            type: 'viewport',
            bbox: [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()],
          }));
        }
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === 'viewport_ack' || payload.type === 'pong') {
            return;
          }
          if (payload.mmsi && payload.latitude !== undefined && payload.longitude !== undefined) {
            setVessels((prev) => {
              const next = new Map(prev);
              const existing = next.get(payload.mmsi);
              next.set(payload.mmsi, {
                ...existing,
                ...payload,
                updated_at: new Date().toISOString(),
              });
              updateMapLayers(next);
              return next;
            });
          }
        } catch (err) {
          console.debug('Invalid WS frame:', err);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      ws?.close();
    };
  }, [updateMapLayers]);

  // Initialize MapLibre
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: STYLE_URL,
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      attributionControl: false,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right');
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left');

    map.on('load', () => {
      // 1. Track Source & Line Layer (Recent Trajectory)
      map.addSource('track-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      map.addLayer({
        id: 'vessel-track-line',
        type: 'line',
        source: 'track-source',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': '#00f0ff',
          'line-width': 2.5,
          'line-dasharray': [2, 2],
        },
      });

      // 1b. Base Maritime Route Layer (Dashed Slate)
      map.addSource('route-base-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'route-base-line',
        type: 'line',
        source: 'route-base-source',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': '#94a3b8',
          'line-width': 2.5,
          'line-dasharray': [3, 2],
        },
      });

      // 1c. Weather-Aware Route Layer (Vibrant Cyan Glow)
      map.addSource('route-opt-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'route-opt-line',
        type: 'line',
        source: 'route-opt-source',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': '#00f0ff',
          'line-width': 3.5,
          'line-opacity': 0.9,
        },
      });

      // 1d. Waypoints Source & Layer (Origin & Destination)
      map.addSource('waypoints-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'waypoints-circle',
        type: 'circle',
        source: 'waypoints-source',
        paint: {
          'circle-radius': 8,
          'circle-color': [
            'case',
            ['==', ['get', 'type'], 'origin'], '#10b981',
            '#ef4444'
          ],
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff',
        },
      });

      // 1e. SAR Scene Footprint Layers
      map.addSource('sar-footprint-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'sar-footprint-fill',
        type: 'fill',
        source: 'sar-footprint-source',
        paint: {
          'fill-color': '#8b5cf6',
          'fill-opacity': 0.15,
        },
      });
      map.addLayer({
        id: 'sar-footprint-line',
        type: 'line',
        source: 'sar-footprint-source',
        paint: {
          'line-color': '#a78bfa',
          'line-width': 2,
          'line-dasharray': [3, 2],
        },
      });

      // 1f. SAR Detections Source & Layers
      map.addSource('sar-detections-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      map.addLayer({
        id: 'sar-detections-pulse',
        type: 'circle',
        source: 'sar-detections-source',
        filter: ['==', ['get', 'isSelected'], true],
        paint: {
          'circle-radius': 16,
          'circle-color': '#c084fc',
          'circle-opacity': 0.45,
          'circle-stroke-width': 2,
          'circle-stroke-color': '#c084fc',
        },
      });
      map.addLayer({
        id: 'sar-detections-circle',
        type: 'circle',
        source: 'sar-detections-source',
        paint: {
          'circle-radius': 8,
          'circle-color': [
            'case',
            ['==', ['get', 'match_status'], 'AIS-MATCHED'], '#10b981',
            '#f59e0b'
          ],
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff',
        },
      });

      // 2. Vessels Source & Layers
      map.addSource('vessels-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      // Outer glow for selected vessel
      map.addLayer({
        id: 'vessels-selected-glow',
        type: 'circle',
        source: 'vessels-source',
        filter: ['==', ['get', 'isSelected'], true],
        paint: {
          'circle-radius': 14,
          'circle-color': '#38bdf8',
          'circle-opacity': 0.35,
          'circle-stroke-width': 2,
          'circle-stroke-color': '#38bdf8',
        },
      });

      // Vessel circle marker
      map.addLayer({
        id: 'vessels-circle',
        type: 'circle',
        source: 'vessels-source',
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            5, 4,
            9, 7,
            14, 11
          ],
          'circle-color': [
            'case',
            ['==', ['get', 'isSelected'], true], '#38bdf8',
            ['>', ['get', 'sog'], 0.5], '#22c55e', // Green if moving
            '#eab308' // Yellow if stationary/anchored
          ],
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#020617',
        },
      });

      // Click on vessel marker
      map.on('click', 'vessels-circle', (e: any) => {
        if (activePickingModeRef.current) return;
        if (!e.features || !e.features[0]) return;
        const clickedMmsi = Number(e.features[0].properties?.mmsi);
        if (clickedMmsi) {
          setSelectedMmsi(clickedMmsi);
        }
      });

      // Click on SAR detection marker
      map.on('click', 'sar-detections-circle', (e: any) => {
        if (!e.features || !e.features[0]) return;
        const detId = e.features[0].properties?.detection_id;
        const found = sarDetectionsRef.current.find((d) => d.detection_id === detId);
        if (found) {
          onSelectDetectionRef.current?.(found);
        }
      });

      map.on('mouseenter', 'sar-detections-circle', () => {
        if (!activePickingModeRef.current) {
          map.getCanvas().style.cursor = 'pointer';
        }
      });

      map.on('mouseleave', 'sar-detections-circle', () => {
        if (!activePickingModeRef.current) {
          map.getCanvas().style.cursor = '';
        }
      });

      // General map click for waypoint selection
      map.on('click', (e: any) => {
        if (activePickingModeRef.current) {
          onMapClickCoordRef.current?.({
            latitude: Number(e.lngLat.lat.toFixed(4)),
            longitude: Number(e.lngLat.lng.toFixed(4)),
          });
        }
      });

      map.on('mouseenter', 'vessels-circle', () => {
        if (!activePickingModeRef.current) {
          map.getCanvas().style.cursor = 'pointer';
        }
      });

      map.on('mouseleave', 'vessels-circle', () => {
        if (!activePickingModeRef.current) {
          map.getCanvas().style.cursor = '';
        }
      });

      // Fetch initial data
      fetchViewportVessels();

      // Resize to ensure MapLibre fills the container after flex layout settles
      map.resize();
    });

    // Debounced moveend listener
    let moveTimeout: ReturnType<typeof setTimeout> | null = null;
    map.on('moveend', () => {
      if (moveTimeout) clearTimeout(moveTimeout);
      moveTimeout = setTimeout(() => {
        fetchViewportVessels();
      }, 400);
    });

    mapRef.current = map;

    // Keep MapLibre in sync if the container is resized (e.g. diagnostics panel toggle)
    let resizeObserver: ResizeObserver | null = null;
    if (mapContainer.current && typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        map.resize();
      });
      resizeObserver.observe(mapContainer.current);
    }

    return () => {
      resizeObserver?.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, [fetchViewportVessels]);

  // Load details and track when selectedMmsi changes
  useEffect(() => {
    if (!selectedMmsi) {
      setSelectedVessel(null);
      updateTrackLayer([]);
      return;
    }

    let active = true;
    setLoadingDetails(true);

    fetchVesselDetails(selectedMmsi)
      .then((details) => {
        if (!active) return;
        setSelectedVessel(details);
        if (details.recent_track && details.recent_track.length > 0) {
          updateTrackLayer(details.recent_track);
        } else {
          updateTrackLayer([]);
        }
      })
      .catch((err) => {
        console.warn('Error fetching vessel details:', err);
      })
      .finally(() => {
        if (active) setLoadingDetails(false);
      });

    return () => {
      active = false;
    };
  }, [selectedMmsi, updateTrackLayer]);

  // Adjust cursor style when picking coordinates on map
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.getCanvas().style.cursor = activePickingMode ? 'crosshair' : '';
  }, [activePickingMode]);

  // Sync Base Route, Weather-Aware Route, and Waypoints to MapLibre sources
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    // 1. Waypoints (Origin & Destination)
    const wpSource = map.getSource('waypoints-source') as maplibregl.GeoJSONSource | undefined;
    if (wpSource) {
      const wpFeatures: any[] = [];
      if (origin) {
        wpFeatures.push({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [origin.longitude, origin.latitude] },
          properties: { type: 'origin', title: 'Origin' },
        });
      }
      if (destination) {
        wpFeatures.push({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [destination.longitude, destination.latitude] },
          properties: { type: 'destination', title: 'Destination' },
        });
      }
      wpSource.setData({
        type: 'FeatureCollection',
        features: wpFeatures,
      });
    }

    // 2. Base Route (Dashed slate)
    const baseSource = map.getSource('route-base-source') as maplibregl.GeoJSONSource | undefined;
    if (baseSource) {
      if (routeResponse?.base_route?.geometry) {
        baseSource.setData({
          type: 'Feature',
          geometry: routeResponse.base_route.geometry as any,
          properties: {},
        });
      } else {
        baseSource.setData({ type: 'FeatureCollection', features: [] });
      }
    }

    // 3. Weather-aware Route (Vibrant cyan)
    const optSource = map.getSource('route-opt-source') as maplibregl.GeoJSONSource | undefined;
    if (optSource) {
      if (routeResponse?.weather_aware_route?.geometry) {
        optSource.setData({
          type: 'Feature',
          geometry: routeResponse.weather_aware_route.geometry as any,
          properties: {},
        });
      } else {
        optSource.setData({ type: 'FeatureCollection', features: [] });
      }
    }

    // Auto-fit bounds if weather-aware route coordinates are present
    if (routeResponse?.weather_aware_route?.geometry?.coordinates?.length) {
      const coords = routeResponse.weather_aware_route.geometry.coordinates;
      const bounds = new maplibregl.LngLatBounds(coords[0] as [number, number], coords[0] as [number, number]);
      coords.forEach((coord: number[]) => {
        bounds.extend(coord as [number, number]);
      });
      map.fitBounds(bounds, { padding: 80, maxZoom: 12 });
    }
  }, [routeResponse, origin, destination]);

  // Sync SAR Scene Footprint & Detections to MapLibre sources
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    // 1. SAR Footprint
    const fpSource = map.getSource('sar-footprint-source') as maplibregl.GeoJSONSource | undefined;
    if (fpSource) {
      if (selectedScene?.footprint) {
        fpSource.setData({
          type: 'Feature',
          geometry: selectedScene.footprint as any,
          properties: { scene_id: selectedScene.scene_id },
        });
      } else {
        fpSource.setData({ type: 'FeatureCollection', features: [] });
      }
    }

    // 2. SAR Detections
    const detSource = map.getSource('sar-detections-source') as maplibregl.GeoJSONSource | undefined;
    if (detSource) {
      const features = (sarDetections || []).map((det) => ({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [det.longitude, det.latitude],
        },
        properties: {
          detection_id: det.detection_id,
          confidence: det.confidence,
          match_status: det.match_status,
          candidate_mmsi: det.candidate_mmsi,
          distance_km: det.distance_km,
          isSelected: selectedDetection?.detection_id === det.detection_id,
        },
      }));
      detSource.setData({
        type: 'FeatureCollection',
        features: features as any,
      });
    }

    // Auto-fit bounds to SAR detections or footprint if available
    if (sarDetections && sarDetections.length > 0) {
      const bounds = new maplibregl.LngLatBounds(
        [sarDetections[0].longitude, sarDetections[0].latitude],
        [sarDetections[0].longitude, sarDetections[0].latitude]
      );
      sarDetections.forEach((d) => {
        bounds.extend([d.longitude, d.latitude]);
      });
      map.fitBounds(bounds, { padding: 80, maxZoom: 13 });
    } else if (selectedScene?.footprint?.coordinates?.[0]?.length) {
      const ring = selectedScene.footprint.coordinates[0];
      const bounds = new maplibregl.LngLatBounds(
        ring[0] as [number, number],
        ring[0] as [number, number]
      );
      ring.forEach((pt: number[]) => {
        bounds.extend(pt as [number, number]);
      });
      map.fitBounds(bounds, { padding: 60, maxZoom: 10 });
    }
  }, [selectedScene, sarDetections, selectedDetection]);

  const statusColors: Record<IngestionStatus, { bg: string; text: string; dot: string }> = {
    LIVE: { bg: '#064e3b', text: '#34d399', dot: '#10b981' },
    STALE: { bg: '#451a03', text: '#fde047', dot: '#eab308' },
    DISCONNECTED: { bg: '#450a0a', text: '#f87171', dot: '#ef4444' },
  };

  const currentStatusColors = statusColors[ingestionStatus];

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />

      {/* Floating Operational Overlay HUD */}
      <div style={{
        position: 'absolute',
        top: '16px',
        left: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
        zIndex: 10
      }}>
        {/* Real-time Status Badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.625rem',
          padding: '0.5rem 0.875rem',
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          backdropFilter: 'blur(6px)',
          borderRadius: '0.5rem',
          border: '1px solid #334155',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
          color: '#f8fafc',
          fontSize: '0.8125rem'
        }}>
          <span style={{
            display: 'inline-block',
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: currentStatusColors.dot,
            boxShadow: `0 0 8px ${currentStatusColors.dot}`
          }} />
          <span style={{ fontWeight: 600, color: currentStatusColors.text }}>
            AIS FEED: {ingestionStatus}
          </span>
          <span style={{ color: '#64748b' }}>|</span>
          <span style={{ color: '#94a3b8' }}>
            Vessels in View: <strong style={{ color: '#f8fafc' }}>{vessels.size}</strong>
          </span>
          {pipelineMetrics && (
            <>
              <span style={{ color: '#64748b' }}>|</span>
              <span style={{ color: '#94a3b8' }}>
                Ingested: <strong style={{ color: '#f8fafc' }}>{pipelineMetrics.messages_received}</strong>
              </span>
            </>
          )}
        </div>

        {/* Informational Banner if Disconnected or Stale */}
        {ingestionStatus !== 'LIVE' && (
          <div style={{
            padding: '0.5rem 0.875rem',
            backgroundColor: 'rgba(30, 41, 59, 0.95)',
            border: `1px solid ${currentStatusColors.dot}`,
            borderRadius: '0.375rem',
            fontSize: '0.75rem',
            color: currentStatusColors.text,
            maxWidth: '380px',
            lineHeight: 1.4
          }}>
            {ingestionStatus === 'DISCONNECTED' ? (
              <span>⚠️ AISStream connection offline or reconnecting in background. No synthetic data will be fabricated.</span>
            ) : (
              <span>⏳ AISStream connected. Waiting for newer position reports in active bounding box...</span>
            )}
          </div>
        )}
      </div>

      {/* Initial Loading Overlay */}
      {initialLoading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          padding: '1rem 1.5rem',
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          borderRadius: '0.5rem',
          border: '1px solid #334155',
          color: '#f8fafc',
          fontSize: '0.875rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          zIndex: 15
        }}>
          <span style={{ fontSize: '1.25rem' }}>🧭</span>
          <span>Loading live maritime traffic for English Channel...</span>
        </div>
      )}

      {/* Empty State Overlay */}
      {!initialLoading && vessels.size === 0 && (
        <div style={{
          position: 'absolute',
          bottom: '24px',
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '0.625rem 1rem',
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          borderRadius: '0.375rem',
          border: '1px solid #334155',
          color: '#94a3b8',
          fontSize: '0.8125rem',
          zIndex: 10
        }}>
          No broadcasting vessels in current viewport. Pan or zoom towards the Dover Strait.
        </div>
      )}

      {/* Selected Vessel Slideout Panel */}
      <VesselDetailPanel
        vessel={selectedVessel}
        loading={loadingDetails}
        onClose={() => {
          setSelectedMmsi(null);
          setSelectedVessel(null);
          updateTrackLayer([]);
        }}
      />
    </div>
  );
};
