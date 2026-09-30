import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';

describe('App Live Vessel Traffic Shell', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(() =>
        Promise.resolve({
          ok: true,
          status: 200,
          json: () =>
            Promise.resolve({
              status: 'ready',
              database: 'ok',
              redis: 'ok',
              timestamp: new Date().toISOString(),
              connection_state: 'connected',
              messages_received: 42,
              normalized_count: 42,
              stream_writes: 42,
              persisted_count: 42,
              last_message_at: new Date().toISOString(),
              total_vessels_in_db: 15,
              bbox: [49.5, -2.0, 51.5, 2.5],
            }),
        })
      )
    );
  });

  it('renders application title and system status', async () => {
    render(<App />);
    expect(screen.getByText('Maritime Operations AI')).toBeInTheDocument();
    expect(screen.getByText('Situational Awareness & Routing Platform')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getAllByText('READY').length).toBeGreaterThan(0);
    });
  });

  it('renders maritime map HUD and allows switching between modules', async () => {
    render(<App />);

    const trafficTab = screen.getByRole('button', { name: /Live Vessel Traffic/i });
    const routingTab = screen.getByRole('button', { name: /AI Weather Routing/i });
    const darkVesselTab = screen.getByRole('button', { name: /Dark-Vessel Intelligence/i });

    expect(trafficTab).toBeInTheDocument();
    expect(routingTab).toBeInTheDocument();
    expect(darkVesselTab).toBeInTheDocument();

    // Default module is traffic with MaritimeMap HUD
    expect(screen.getByText(/AIS FEED:/i)).toBeInTheDocument();
    expect(screen.getByText(/Vessels in View:/i)).toBeInTheDocument();

    // Switch to routing module (renders RoutingControlPanel on unified map)
    fireEvent.click(routingTab);
    expect(screen.getByText('AI Weather-Aware Routing')).toBeInTheDocument();
    expect(screen.getByText('Voyage Waypoints')).toBeInTheDocument();

    // Switch to dark-vessel module
    fireEvent.click(darkVesselTab);
    expect(screen.getByText(/Observation Search Area/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Search Sentinel-1 Scenes/i })).toBeInTheDocument();

    // Switch back to traffic module
    fireEvent.click(trafficTab);
    expect(screen.getByText(/AIS FEED:/i)).toBeInTheDocument();
  });
});
