import React from 'react';
import { ActiveModule } from '../types';

interface NavigationProps {
  activeModule: ActiveModule;
  onSelectModule: (module: ActiveModule) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ activeModule, onSelectModule }) => {
  const tabs: { id: ActiveModule; label: string; icon: string }[] = [
    { id: 'traffic', label: 'Live Vessel Traffic', icon: '🚢' },
    { id: 'routing', label: 'AI Weather Routing', icon: '🌊' },
    { id: 'dark-vessel', label: 'Dark-Vessel Intelligence', icon: '🛰️' },
  ];

  return (
    <nav style={{
      display: 'flex',
      gap: '0.5rem',
      padding: '0.5rem 1.5rem',
      backgroundColor: '#1e293b',
      borderBottom: '1px solid #334155'
    }}>
      {tabs.map((tab) => {
        const isActive = activeModule === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onSelectModule(tab.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.375rem',
              padding: '0.5rem 1rem',
              borderRadius: '0.375rem',
              border: 'none',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: 500,
              backgroundColor: isActive ? '#3b82f6' : 'transparent',
              color: isActive ? '#ffffff' : '#94a3b8',
              transition: 'background-color 0.15s ease, color 0.15s ease'
            }}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        );
      })}
    </nav>
  );
};
