import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Navigation } from './components/Navigation';
import { HealthStatus } from './components/HealthStatus';
import { MaritimeMap } from './components/MaritimeMap';
import { RoutingControlPanel } from './components/RoutingControlPanel';
import { DarkVesselControlPanel } from './components/DarkVesselControlPanel';
import { fetchHealthReady } from './services/api';
import { ActiveModule, HealthReadyResponse, IngestionStatus } from './types';
import { RouteOptimizationResponse, GeoCoordinate } from './services/routingApi';
import { SARSceneSummary, SARDetectionItem } from './services/sarApi';

export const App: React.FC = () => {
  const [activeModule, setActiveModule] = useState<ActiveModule>('traffic');
  const [healthData, setHealthData] = useState<HealthReadyResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus>('DISCONNECTED');
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(false);

  // Routing State
  const [routeResponse, setRouteResponse] = useState<RouteOptimizationResponse | null>(null);
  const [origin, setOrigin] = useState<GeoCoordinate | null>({ latitude: 48.38, longitude: -4.5 }); // Brest
  const [destination, setDestination] = useState<GeoCoordinate | null>({ latitude: 51.12, longitude: 1.3 }); // Dover
  const [activePickingMode, setActivePickingMode] = useState<"origin" | "destination" | null>(null);

  // Dark-Vessel SAR State
  const [selectedSARScene, setSelectedSARScene] = useState<SARSceneSummary | null>(null);
  const [sarDetections, setSARDetections] = useState<SARDetectionItem[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<SARDetectionItem | null>(null);

  const handleMapClickCoord = useCallback((coord: GeoCoordinate) => {
    if (activePickingMode === "origin") {
      setOrigin(coord);
      setActivePickingMode(null);
    } else if (activePickingMode === "destination") {
      setDestination(coord);
      setActivePickingMode(null);
    }
  }, [activePickingMode]);

  const checkHealth = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchHealthReady();
      setHealthData(data);
      setHealthError(null);
    } catch (err: unknown) {
      setHealthError(err instanceof Error ? err.message : 'Unknown error');
      setHealthData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return (
    <div style={{ height: '100%', backgroundColor: '#020617', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Header
        systemStatus={healthData?.status || 'starting'}
      />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#1e293b', borderBottom: '1px solid #334155', flexShrink: 0 }}>
        <Navigation activeModule={activeModule} onSelectModule={setActiveModule} />
        <div style={{ paddingRight: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={() => setShowDiagnostics((prev) => !prev)}
            style={{
              padding: '0.35rem 0.65rem',
              borderRadius: '0.25rem',
              backgroundColor: showDiagnostics ? '#334155' : 'transparent',
              color: '#94a3b8',
              border: '1px solid #475569',
              fontSize: '0.75rem',
              cursor: 'pointer',
            }}
          >
            {showDiagnostics ? 'Hide Diagnostics' : 'Show Diagnostics'}
          </button>
        </div>
      </div>

      {showDiagnostics && (
        <div style={{ padding: '1rem 1.5rem 0 1.5rem', maxWidth: '1200px', margin: '0 auto', width: '100%', boxSizing: 'border-box', flexShrink: 0 }}>
          <HealthStatus
            healthData={healthData}
            error={healthError}
            onRefresh={checkHealth}
            loading={loading}
          />
        </div>
      )}

      {/* Map shell: fills all remaining vertical space, panels float absolutely over the map */}
      <main style={{ flex: 1, position: 'relative', overflow: 'hidden', minHeight: 0 }}>
        <MaritimeMap
          onStatusChange={setIngestionStatus}
          routeResponse={routeResponse}
          origin={origin}
          destination={destination}
          activePickingMode={activePickingMode}
          onMapClickCoord={handleMapClickCoord}
          selectedScene={selectedSARScene}
          sarDetections={sarDetections}
          selectedDetection={selectedDetection}
          onSelectDetection={setSelectedDetection}
        />
        {activeModule === 'routing' && (
          <RoutingControlPanel
            onRouteCalculated={setRouteResponse}
            activePickingMode={activePickingMode}
            onSetPickingMode={setActivePickingMode}
            origin={origin}
            destination={destination}
            onUpdateOrigin={setOrigin}
            onUpdateDestination={setDestination}
          />
        )}
        {activeModule === 'dark-vessel' && (
          <DarkVesselControlPanel
            onSceneSelect={setSelectedSARScene}
            onDetectionsLoaded={setSARDetections}
            onSelectDetection={setSelectedDetection}
            selectedDetection={selectedDetection}
          />
        )}
      </main>

      <footer style={{
        padding: '0.5rem 1rem',
        textAlign: 'center',
        fontSize: '0.75rem',
        color: '#64748b',
        borderTop: '1px solid #1e293b',
        backgroundColor: '#0f172a',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexShrink: 0,
      }}>
        <span>Maritime Operations AI &bull; English Channel / Dover Strait AIS Corridor</span>
        <span>Zero Out-Of-Pocket Architecture &bull; PostgreSQL/PostGIS &bull; Redis 7 Streams</span>
      </footer>
    </div>
  );
};

export default App;
