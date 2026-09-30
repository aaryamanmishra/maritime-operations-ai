import React from 'react';
import { HealthReadyResponse } from '../types';

interface HealthStatusProps {
  healthData: HealthReadyResponse | null;
  error: string | null;
  onRefresh: () => void;
  loading: boolean;
}

export const HealthStatus: React.FC<HealthStatusProps> = ({ healthData, error, onRefresh, loading }) => {
  return (
    <div style={{
      backgroundColor: '#1e293b',
      borderRadius: '0.5rem',
      padding: '1.25rem',
      border: '1px solid #334155',
      marginBottom: '1rem',
      color: '#f8fafc'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>System Foundation Status</h3>
        <button
          onClick={onRefresh}
          disabled={loading}
          style={{
            padding: '0.375rem 0.75rem',
            borderRadius: '0.25rem',
            backgroundColor: '#334155',
            color: '#f8fafc',
            border: 'none',
            fontSize: '0.75rem',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Checking...' : 'Refresh Status'}
        </button>
      </div>

      {error ? (
        <div style={{ color: '#f87171', fontSize: '0.875rem' }}>
          ⚠️ Backend unreachable: {error}
        </div>
      ) : healthData ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem' }}>
          <div style={{ backgroundColor: '#0f172a', padding: '0.75rem', borderRadius: '0.375rem' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>API Status</span>
            <div style={{ fontSize: '0.875rem', fontWeight: 600, color: healthData.status === 'ready' ? '#4ade80' : '#facc15' }}>
              {healthData.status.toUpperCase()}
            </div>
          </div>
          <div style={{ backgroundColor: '#0f172a', padding: '0.75rem', borderRadius: '0.375rem' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>PostgreSQL / PostGIS</span>
            <div style={{ fontSize: '0.875rem', fontWeight: 600, color: healthData.database === 'ok' ? '#4ade80' : '#f87171' }}>
              {healthData.database.toUpperCase()}
            </div>
          </div>
          <div style={{ backgroundColor: '#0f172a', padding: '0.75rem', borderRadius: '0.375rem' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Redis Broker/Cache</span>
            <div style={{ fontSize: '0.875rem', fontWeight: 600, color: healthData.redis === 'ok' ? '#4ade80' : '#f87171' }}>
              {healthData.redis.toUpperCase()}
            </div>
          </div>
          <div style={{ backgroundColor: '#0f172a', padding: '0.75rem', borderRadius: '0.375rem' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Phase</span>
            <div style={{ fontSize: '0.875rem', fontWeight: 600, color: '#38bdf8' }}>
              PHASE 1 (FOUNDATION)
            </div>
          </div>
        </div>
      ) : (
        <div style={{ fontSize: '0.875rem', color: '#94a3b8' }}>Loading system health...</div>
      )}
    </div>
  );
};
