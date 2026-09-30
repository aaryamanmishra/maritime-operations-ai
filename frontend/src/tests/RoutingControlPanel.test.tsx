import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RoutingControlPanel } from '../components/RoutingControlPanel';
import * as routingApi from '../services/routingApi';

vi.mock('../services/routingApi', async () => {
  const actual = await vi.importActual('../services/routingApi');
  return {
    ...actual,
    fetchVesselDefaults: vi.fn(),
    optimizeRoute: vi.fn(),
  };
});

describe('RoutingControlPanel Component', () => {
  const mockOnRouteCalculated = vi.fn();
  const mockOnSetPickingMode = vi.fn();
  const mockOnUpdateOrigin = vi.fn();
  const mockOnUpdateDestination = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (routingApi.fetchVesselDefaults as any).mockResolvedValue({
      vessel_type: 'tanker',
      length_m: 250,
      beam_m: 44,
      draft_m: 14.5,
      design_speed_kn: 13.5,
      displacement_t: 110000,
      propulsion_power_kw: 16000,
    });
  });

  it('renders routing form fields and disclaimers', () => {
    render(
      <RoutingControlPanel
        onRouteCalculated={mockOnRouteCalculated}
        activePickingMode={null}
        onSetPickingMode={mockOnSetPickingMode}
        origin={{ latitude: 48.38, longitude: -4.5 }}
        destination={{ latitude: 51.12, longitude: 1.3 }}
        onUpdateOrigin={mockOnUpdateOrigin}
        onUpdateDestination={mockOnUpdateDestination}
      />
    );

    expect(screen.getByText('AI Weather-Aware Routing')).toBeInTheDocument();
    expect(screen.getByText('Voyage Waypoints')).toBeInTheDocument();
    expect(screen.getByText(/48.380°, -4.500°/i)).toBeInTheDocument();
    expect(screen.getByText(/51.120°, 1.300°/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Calculate Weather-Aware Route/i })).toBeInTheDocument();
  });

  it('triggers coordinate picking mode when Set on Map clicked', () => {
    render(
      <RoutingControlPanel
        onRouteCalculated={mockOnRouteCalculated}
        activePickingMode={null}
        onSetPickingMode={mockOnSetPickingMode}
        origin={{ latitude: 48.38, longitude: -4.5 }}
        destination={{ latitude: 51.12, longitude: 1.3 }}
        onUpdateOrigin={mockOnUpdateOrigin}
        onUpdateDestination={mockOnUpdateDestination}
      />
    );

    const setButtons = screen.getAllByRole('button', { name: /Set on Map/i });
    fireEvent.click(setButtons[0]);
    expect(mockOnSetPickingMode).toHaveBeenCalledWith('origin');
  });

  it('submits optimization request and displays estimate labels', async () => {
    const mockResponse: routingApi.RouteOptimizationResponse = {
      request: {} as any,
      base_route: {
        route_type: 'base_maritime',
        geometry: { type: 'LineString', coordinates: [[-4.5, 48.38], [1.3, 51.12]] },
        summary: {
          total_distance_nm: 329.03,
          estimated_duration_hours: 23.71,
          departure_time: '2026-09-30T12:00:00Z',
          estimated_eta: '2026-10-01T11:42:00Z',
          mean_speed_kn: 13.88,
          mean_speed_loss_kn: 0.62,
          total_estimated_energy_kwh: 105000,
          total_estimated_fuel_tonnes: 30.85,
          total_estimated_cost_usd: 19127,
          max_wave_height_m: 2.1,
          max_wind_speed_mps: 12.0,
          avg_wave_height_m: 1.5,
          weather_exposure_level: 'moderate',
        },
        segments: [],
      },
      weather_aware_route: {
        route_type: 'weather_aware',
        geometry: { type: 'LineString', coordinates: [[-4.5, 48.38], [1.3, 51.12]] },
        summary: {
          total_distance_nm: 329.03,
          estimated_duration_hours: 23.71,
          departure_time: '2026-09-30T12:00:00Z',
          estimated_eta: '2026-10-01T11:42:00Z',
          mean_speed_kn: 13.88,
          mean_speed_loss_kn: 0.62,
          total_estimated_energy_kwh: 105000,
          total_estimated_fuel_tonnes: 30.85,
          total_estimated_cost_usd: 19127,
          max_wave_height_m: 2.1,
          max_wind_speed_mps: 12.0,
          avg_wave_height_m: 1.5,
          weather_exposure_level: 'moderate',
        },
        segments: [],
      },
      route_diverged: false,
      summary_comparison: {
        distance_difference_nm: 0,
        duration_difference_hours: 0,
        estimated_fuel_difference_tonnes: 0,
        estimated_cost_difference_usd: 0,
        base_max_wave_height_m: 2.1,
        optimized_max_wave_height_m: 2.1,
        wave_exposure_reduction_m: 0,
      },
      model_version: 'tft-maris-noaa-v1',
      weather_provider: 'Open-Meteo Marine (ECMWF IFS / GFS Waves)',
      disclaimer: 'All fuel, energy, and speed loss values are ESTIMATES based on naval architecture and machine learning models.',
    };

    (routingApi.optimizeRoute as any).mockResolvedValue(mockResponse);

    render(
      <RoutingControlPanel
        onRouteCalculated={mockOnRouteCalculated}
        activePickingMode={null}
        onSetPickingMode={mockOnSetPickingMode}
        origin={{ latitude: 48.38, longitude: -4.5 }}
        destination={{ latitude: 51.12, longitude: 1.3 }}
        onUpdateOrigin={mockOnUpdateOrigin}
        onUpdateDestination={mockOnUpdateDestination}
      />
    );

    const calcBtn = screen.getByRole('button', { name: /Calculate Weather-Aware Route/i });
    fireEvent.click(calcBtn);

    await waitFor(() => {
      expect(mockOnRouteCalculated).toHaveBeenCalledWith(mockResponse);
      expect(screen.getByText(/Route Comparison/i)).toBeInTheDocument();
      expect(screen.getByText(/All fuel, energy, and speed loss values are ESTIMATES/i)).toBeInTheDocument();
    });
  });
});
