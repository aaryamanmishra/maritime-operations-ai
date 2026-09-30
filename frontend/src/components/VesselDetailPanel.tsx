import React from 'react';
import { VesselDetails } from '../types';

interface VesselDetailPanelProps {
  vessel: VesselDetails | null;
  loading: boolean;
  onClose: () => void;
}

export const VesselDetailPanel: React.FC<VesselDetailPanelProps> = ({ vessel, loading, onClose }) => {
  if (!vessel && !loading) {
    return null;
  }

  const formatValue = (val: string | number | null | undefined, unit: string = ''): string => {
    if (val === null || val === undefined || val === '') {
      return 'Unavailable';
    }
    return `${val}${unit}`;
  };

  const getNavStatusText = (status: number | null): string => {
    if (status === null || status === undefined) return 'Unavailable';
    const statusMap: Record<number, string> = {
      0: 'Under way using engine',
      1: 'At anchor',
      2: 'Not under command',
      3: 'Restricted manoeuvrability',
      4: 'Constrained by draft',
      5: 'Moored',
      6: 'Aground',
      7: 'Engaged in fishing',
      8: 'Under way sailing',
      14: 'AIS-SART active',
      15: 'Undefined / Default',
    };
    return statusMap[status] || `Status code ${status}`;
  };

  const getShipTypeText = (type: number | null): string => {
    if (type === null || type === undefined) return 'Unavailable';
    if (type >= 70 && type <= 79) return `Cargo (${type})`;
    if (type >= 80 && type <= 89) return `Tanker (${type})`;
    if (type >= 60 && type <= 69) return `Passenger (${type})`;
    if (type === 30) return `Fishing (${type})`;
    if (type === 31 || type === 32) return `Towing / Tug (${type})`;
    if (type >= 50 && type <= 59) return `Special Craft (${type})`;
    return `Type code ${type}`;
  };

  return (
    <aside style={{
      position: 'absolute',
      top: '70px',
      right: '16px',
      width: '360px',
      maxHeight: 'calc(100vh - 90px)',
      overflowY: 'auto',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      backdropFilter: 'blur(8px)',
      border: '1px solid #334155',
      borderRadius: '0.75rem',
      boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5)',
      color: '#f8fafc',
      zIndex: 20,
      padding: '1.25rem',
      boxSizing: 'border-box'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem', borderBottom: '1px solid #1e293b', paddingBottom: '0.75rem' }}>
        <div>
          <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8' }}>
            Vessel Particulars
          </span>
          <h2 style={{ margin: '0.25rem 0 0 0', fontSize: '1.25rem', fontWeight: 600 }}>
            {vessel?.name || 'Unknown Vessel'}
          </h2>
          <div style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>
            MMSI: <span style={{ color: '#f8fafc', fontFamily: 'monospace' }}>{vessel?.mmsi}</span>
          </div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close panel"
          style={{
            background: 'none',
            border: 'none',
            color: '#94a3b8',
            fontSize: '1.25rem',
            cursor: 'pointer',
            padding: '0.25rem',
            lineHeight: 1
          }}
        >
          &times;
        </button>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '2rem 0', color: '#94a3b8' }}>
          Loading vessel telemetry...
        </div>
      ) : vessel ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem', fontSize: '0.875rem' }}>
          {/* Kinematics Card */}
          <div style={{ backgroundColor: '#0f172a', padding: '0.75rem', borderRadius: '0.5rem', border: '1px solid #1e293b' }}>
            <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Live Kinematics</span>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.375rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Speed (SOG)</span>
                <div style={{ fontWeight: 600, color: '#38bdf8' }}>
                  {formatValue(vessel.sog, ' kts')}
                </div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Course (COG)</span>
                <div style={{ fontWeight: 600, color: '#f8fafc' }}>
                  {formatValue(vessel.cog, '°')}
                </div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Heading</span>
                <div style={{ fontWeight: 600, color: '#f8fafc' }}>
                  {formatValue(vessel.heading, '°')}
                </div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Status</span>
                <div style={{ fontWeight: 500, fontSize: '0.75rem', color: '#f8fafc' }}>
                  {getNavStatusText(vessel.nav_status)}
                </div>
              </div>
            </div>
            <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #1e293b', fontSize: '0.75rem', color: '#94a3b8' }}>
              Coordinates: <span style={{ fontFamily: 'monospace', color: '#f8fafc' }}>{vessel.latitude.toFixed(4)}°, {vessel.longitude.toFixed(4)}°</span>
            </div>
          </div>

          {/* Static Metadata */}
          <div style={{ backgroundColor: '#0f172a', padding: '0.75rem', borderRadius: '0.5rem', border: '1px solid #1e293b' }}>
            <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Voyage & Dimensions</span>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.375rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Ship Type</span>
                <div style={{ fontWeight: 500 }}>{getShipTypeText(vessel.ship_type)}</div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>IMO</span>
                <div style={{ fontFamily: 'monospace' }}>{formatValue(vessel.imo)}</div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Callsign</span>
                <div style={{ fontFamily: 'monospace' }}>{formatValue(vessel.callsign)}</div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Destination</span>
                <div style={{ fontWeight: 500 }}>{formatValue(vessel.destination)}</div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Dimensions</span>
                <div>{vessel.length && vessel.beam ? `${vessel.length}m × ${vessel.beam}m` : 'Unavailable'}</div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Draught</span>
                <div>{formatValue(vessel.draught, ' m')}</div>
              </div>
            </div>
          </div>

          {/* Track points */}
          <div style={{ backgroundColor: '#0f172a', padding: '0.75rem', borderRadius: '0.5rem', border: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Historical Track</span>
              <span style={{ fontSize: '0.75rem', color: '#38bdf8', fontWeight: 500 }}>
                {vessel.recent_track?.length || 0} recorded points
              </span>
            </div>
            <p style={{ margin: '0.375rem 0 0 0', fontSize: '0.75rem', color: '#94a3b8' }}>
              Persisted trajectory displayed as cyan trail on the vector map.
            </p>
          </div>

          {/* Timestamp footer */}
          <div style={{ fontSize: '0.75rem', color: '#64748b', textAlign: 'right' }}>
            Last position report: {new Date(vessel.timestamp).toLocaleTimeString()}
          </div>
        </div>
      ) : null}
    </aside>
  );
};
