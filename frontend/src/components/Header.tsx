import React from 'react';

interface HeaderProps {
  systemStatus: string;
}

export const Header: React.FC<HeaderProps> = ({ systemStatus }) => {
  const isReady = systemStatus === 'ready';

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0.75rem 1.5rem',
      backgroundColor: '#0f172a',
      color: '#f8fafc',
      borderBottom: '1px solid #1e293b'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <span style={{ fontSize: '1.25rem' }}>⚓</span>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 600 }}>Maritime Operations AI</h1>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Situational Awareness & Routing Platform</span>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
        <span style={{
          display: 'inline-block',
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: isReady ? '#22c55e' : '#eab308'
        }} />
        <span style={{ color: isReady ? '#86efac' : '#fde047' }}>
          {systemStatus.toUpperCase()}
        </span>
      </div>
    </header>
  );
};
