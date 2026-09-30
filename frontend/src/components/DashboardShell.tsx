import React from 'react';
import { ActiveModule } from '../types';

interface DashboardShellProps {
  activeModule: ActiveModule;
}

export const DashboardShell: React.FC<DashboardShellProps> = ({ activeModule }) => {
  const moduleInfo: Record<ActiveModule, { title: string; description: string; readiness: string }> = {
    traffic: {
      title: 'Live Vessel Traffic Module',
      description: 'Interactive maritime map, real AIS vessel position ingestion, and kinematic tracking.',
      readiness: 'Foundation ready. AISStream adapter & MapLibre map engine queued for Phase 2.',
    },
    routing: {
      title: 'AI Weather-Aware Routing Module',
      description: 'Eurostat/SeaRoute network optimization, Open-Meteo marine grids, and DL performance surrogate.',
      readiness: 'Foundation ready. Navigational graph & hydrodynamic models queued for Phase 3.',
    },
    'dark-vessel': {
      title: 'Dark-Vessel Intelligence Module',
      description: 'Copernicus Sentinel-1 SAR ship detection, spatiotemporal AIS correlation, and unmatched contact analysis.',
      readiness: 'Foundation ready. STAC scene discovery & modern SAR detector pipeline queued for Phase 4.',
    },
  };

  const active = moduleInfo[activeModule];

  return (
    <div style={{
      backgroundColor: '#0f172a',
      borderRadius: '0.5rem',
      padding: '2rem',
      border: '1px solid #1e293b',
      color: '#f8fafc',
      minHeight: '300px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      textAlign: 'center'
    }}>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', fontWeight: 600 }}>{active.title}</h2>
      <p style={{ maxWidth: '600px', color: '#94a3b8', marginBottom: '1.5rem', lineHeight: 1.5 }}>
        {active.description}
      </p>
      <div style={{
        padding: '0.5rem 1rem',
        borderRadius: '0.375rem',
        backgroundColor: '#1e293b',
        border: '1px solid #334155',
        fontSize: '0.875rem',
        color: '#38bdf8'
      }}>
        ℹ️ {active.readiness}
      </div>
    </div>
  );
};
