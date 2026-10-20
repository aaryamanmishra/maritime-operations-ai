import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DarkVesselControlPanel } from '../components/DarkVesselControlPanel';
import { SARDetectionItem, SARJobResponse, SARSceneSummary } from '../services/sarApi';

describe('DarkVesselControlPanel Geographic Context Integration', () => {
  const mockOnSceneSelect = vi.fn();
  const mockOnDetectionsLoaded = vi.fn();
  const mockOnSelectDetection = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockScene: SARSceneSummary = {
    scene_id: 'S1A_IW_SLC__1SDV_20241015T103000_258592',
    acquisition_time: '2024-10-15T10:30:00Z',
    platform: 'Sentinel-1A',
    orbit_pass: 'ASCENDING',
    footprint: {
      type: 'Polygon',
      coordinates: [[[90.0, 39.0], [90.5, 39.0], [90.5, 39.5], [90.0, 39.5], [90.0, 39.0]]],
    },
    polarization: 'VV+VH',
    processing_level: 'LEVEL1_SLC',
    modality: 'SLC',
    source: 'Copernicus CDSE',
  };

  it('correctly handles the Lop Nur hard-negative case: 0 operational vessels and clutter candidate relegated to rejected diagnostic view', async () => {
    // Lop Nur hard-negative detection: job sar-job-8dc1986a48
    const lopNurClutterDetection: SARDetectionItem = {
      detection_id: 'sar-det-d7ad02e15c27',
      scene_id: mockScene.scene_id,
      acquisition_time: '2024-10-15T10:30:00Z',
      latitude: 39.42758,
      longitude: 90.09789,
      pixel_bbox: { ymin: 183.3, xmin: 1159.2, ymax: 193.1, xmax: 1177.9 },
      confidence: 0.2539,
      model_version: 'yolo26s-opensar-slc-v1',
      modality: 'SLC',
      length_m: 161.1,
      match_status: 'AIS-UNMATCHED',
      surface_context: 'TERRESTRIAL_CLUTTER',
      is_operational_vessel: false,
      provenance: {
        geolocation_method: 'piecewise_gcp_grid',
        range_pixel_spacing_m: 2.329562,
        azimuth_pixel_spacing_m: 13.96549,
        geographic_context: {
          status: 'TERRESTRIAL_CLUTTER',
          reason: 'Candidate is located 1848.7 km inland from coastline on continental landmass; flagged as terrestrial radar clutter.',
          is_operational_vessel: false,
          distance_to_coast_km: 1848.72,
          is_on_land: true,
          data_source: 'Natural Earth 1:110m Physical Land',
        },
      },
    };

    const mockJobResponse: SARJobResponse = {
      job_id: 'sar-job-8dc1986a48',
      scene_id: mockScene.scene_id,
      status: 'COMPLETE',
      modality: 'SLC',
      progress_percent: 100,
      message: 'Analysis complete: 0 operational vessels detected (0 AIS-MATCHED, 1 AIS-UNMATCHED, 1 terrestrial clutter candidates flagged)',
      detections_count: 0,
      matched_count: 0,
      unmatched_count: 0,
      created_at: '2024-10-15T10:35:00Z',
      detections: [lopNurClutterDetection],
    };

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (url.includes('/api/v1/sar/search')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve([mockScene]),
          });
        }
        if (url.includes('/api/v1/sar/jobs/') && !url.endsWith('/jobs')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve(mockJobResponse),
          });
        }
        if (url.endsWith('/api/v1/sar/jobs')) {
          return Promise.resolve({
            ok: true,
            status: 202,
            json: () => Promise.resolve(mockJobResponse),
          });
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve([]),
        });
      })
    );

    const { rerender } = render(
      <DarkVesselControlPanel
        onSceneSelect={mockOnSceneSelect}
        onDetectionsLoaded={mockOnDetectionsLoaded}
        onSelectDetection={mockOnSelectDetection}
        selectedDetection={null}
      />
    );

    // Switch to SLC modality and search
    const slcButton = screen.getByRole('button', { name: 'SLC' });
    fireEvent.click(slcButton);

    const searchButton = screen.getByRole('button', { name: /Search Sentinel-1 Scenes/i });
    fireEvent.click(searchButton);

    // Wait for scenes list and click on scene
    await waitFor(() => {
      expect(screen.getByText('Sentinel-1A')).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText('Sentinel-1A'));

    // Click analyze scene
    const analyzeButton = screen.getByRole('button', { name: /Analyze Scene & Correlate AIS/i });
    fireEvent.click(analyzeButton);

    // Wait for analysis to complete
    await waitFor(() => {
      expect(screen.getByTestId('operational-vessels-count')).toHaveTextContent('0 Operational Vessels');
    });

    // Verify operational count is 0 and clutter candidate is NOT counted as an operational vessel
    expect(screen.getByTestId('matched-count')).toHaveTextContent('0');
    expect(screen.getByTestId('unmatched-count')).toHaveTextContent('0');
    expect(screen.getByTestId('clutter-count')).toHaveTextContent('1');

    // Verify banner explains that 0 operational vessels were detected and candidate was rejected as clutter
    expect(screen.getByText('0 Operational Vessels Detected')).toBeInTheDocument();
    expect(screen.getByText(/1 candidate\(s\) rejected as terrestrial radar clutter/i)).toBeInTheDocument();

    // Verify rejected candidate appears under the rejected terrestrial clutter diagnostic view
    expect(screen.getByText(/Rejected Terrestrial Clutter \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText('REJECTED CLUTTER')).toBeInTheDocument();
    expect(screen.getByText(/1848.7 km inland/i)).toBeInTheDocument();

    // Re-render with selectedDetection set to the clutter candidate to verify provenance panel
    rerender(
      <DarkVesselControlPanel
        onSceneSelect={mockOnSceneSelect}
        onDetectionsLoaded={mockOnDetectionsLoaded}
        onSelectDetection={mockOnSelectDetection}
        selectedDetection={lopNurClutterDetection}
      />
    );

    // Verify detailed provenance displays all required diagnostic fields
    const provenancePanel = screen.getByTestId('detection-provenance-panel');
    expect(provenancePanel).toBeInTheDocument();
    expect(screen.getByText('REJECTED (Terrestrial Clutter)')).toBeInTheDocument();
    expect(screen.getByText('TERRESTRIAL_CLUTTER')).toBeInTheDocument();
    expect(screen.getByText(/1848.7 km inland from coastline/i)).toBeInTheDocument();
    expect(screen.getByText(/piecewise_gcp_grid/i)).toBeInTheDocument();
    expect(screen.getByText(/Rg 2.33 m • Az 13.97 m/i)).toBeInTheDocument();
    expect(screen.getByText(/Natural Earth 1:110m Physical Land/i)).toBeInTheDocument();
    expect(screen.getByText(/25.4%/i)).toBeInTheDocument();
  });

  it('preserves dark vessels: AIS-unmatched detection over plausible water is counted as operational dark vessel and never rejected as clutter', async () => {
    const darkVesselDetection: SARDetectionItem = {
      detection_id: 'sar-det-dark-vessel-01',
      scene_id: mockScene.scene_id,
      acquisition_time: '2024-10-15T10:30:00Z',
      latitude: 50.1234,
      longitude: -1.2345,
      confidence: 0.88,
      model_version: 'yolo26s-opensar-slc-v1',
      modality: 'SLC',
      match_status: 'AIS-UNMATCHED',
      surface_context: 'PLAUSIBLE_WATER',
      is_operational_vessel: true,
      provenance: {
        geolocation_method: 'piecewise_gcp_grid',
        range_pixel_spacing_m: 2.329562,
        azimuth_pixel_spacing_m: 13.96549,
        geographic_context: {
          status: 'PLAUSIBLE_WATER',
          reason: 'Candidate is located in open maritime waters (distance to coast: 28.4 km).',
          is_operational_vessel: true,
          distance_to_coast_km: 28.4,
          is_on_land: false,
        },
      },
    };

    const mockJobResponse: SARJobResponse = {
      job_id: 'sar-job-dark-vessel',
      scene_id: mockScene.scene_id,
      status: 'COMPLETE',
      modality: 'SLC',
      progress_percent: 100,
      message: 'Analysis complete: 1 operational vessels detected (0 AIS-MATCHED, 1 AIS-UNMATCHED, 0 terrestrial clutter candidates flagged)',
      detections_count: 1,
      matched_count: 0,
      unmatched_count: 1,
      created_at: '2024-10-15T10:35:00Z',
      detections: [darkVesselDetection],
    };

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (url.includes('/api/v1/sar/search')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve([mockScene]),
          });
        }
        if (url.includes('/api/v1/sar/jobs/')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve(mockJobResponse),
          });
        }
        if (url.endsWith('/api/v1/sar/jobs')) {
          return Promise.resolve({
            ok: true,
            status: 202,
            json: () => Promise.resolve(mockJobResponse),
          });
        }
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve([]) });
      })
    );

    render(
      <DarkVesselControlPanel
        onSceneSelect={mockOnSceneSelect}
        onDetectionsLoaded={mockOnDetectionsLoaded}
        onSelectDetection={mockOnSelectDetection}
        selectedDetection={darkVesselDetection}
      />
    );

    // Search and analyze scene
    fireEvent.click(screen.getByRole('button', { name: /Search Sentinel-1 Scenes/i }));
    await waitFor(() => screen.getByText('Sentinel-1A'));
    fireEvent.click(screen.getByText('Sentinel-1A'));
    fireEvent.click(screen.getByRole('button', { name: /Analyze Scene & Correlate AIS/i }));

    await waitFor(() => {
      expect(screen.getByTestId('operational-vessels-count')).toHaveTextContent('1 Operational Vessels');
    });

    // Dark vessel count is 1, clutter count is 0 (not rendered)
    expect(screen.getByTestId('unmatched-count')).toHaveTextContent('1');
    expect(screen.getByTestId('matched-count')).toHaveTextContent('0');
    expect(screen.queryByTestId('clutter-count')).toBeNull();

    // Appears in operational surface contacts with "Dark Vessel Target"
    expect(screen.getByText(/Dark Vessel Target/i)).toBeInTheDocument();
    expect(screen.getByText('OPERATIONAL VESSEL')).toBeInTheDocument();
    expect(screen.getByText('PLAUSIBLE_WATER')).toBeInTheDocument();

    // Verify ethical disclaimer is intact
    expect(screen.getByText(/Lack of AIS broadcast is never treated as proof of clutter/i)).toBeInTheDocument();
  });

  it('displays ambiguous coastal candidate with nearshore uncertainty visibly communicated and counted as operational', async () => {
    const ambiguousCoastalDetection: SARDetectionItem = {
      detection_id: 'sar-det-coastal-01',
      scene_id: mockScene.scene_id,
      acquisition_time: '2024-10-15T10:30:00Z',
      latitude: 50.85,
      longitude: 1.45,
      confidence: 0.72,
      model_version: 'yolo26s-opensar-slc-v1',
      modality: 'SLC',
      match_status: 'AIS-MATCHED',
      candidate_mmsi: 227123456,
      distance_km: 0.35,
      time_difference_seconds: 45,
      surface_context: 'AMBIGUOUS_COASTAL',
      is_operational_vessel: true,
      provenance: {
        geolocation_method: 'piecewise_gcp_grid',
        range_pixel_spacing_m: 2.329562,
        azimuth_pixel_spacing_m: 13.96549,
        geographic_context: {
          status: 'AMBIGUOUS_COASTAL',
          reason: 'Candidate is within 4.2 km of coastline; classified as ambiguous nearshore boundary.',
          is_operational_vessel: true,
          distance_to_coast_km: 4.2,
          is_on_land: false,
        },
      },
    };

    const mockJobResponse: SARJobResponse = {
      job_id: 'sar-job-coastal',
      scene_id: mockScene.scene_id,
      status: 'COMPLETE',
      modality: 'SLC',
      progress_percent: 100,
      message: 'Analysis complete: 1 operational vessels detected',
      detections_count: 1,
      matched_count: 1,
      unmatched_count: 0,
      created_at: '2024-10-15T10:35:00Z',
      detections: [ambiguousCoastalDetection],
    };

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (url.includes('/api/v1/sar/search')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve([mockScene]),
          });
        }
        if (url.includes('/api/v1/sar/jobs/')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve(mockJobResponse),
          });
        }
        if (url.endsWith('/api/v1/sar/jobs')) {
          return Promise.resolve({
            ok: true,
            status: 202,
            json: () => Promise.resolve(mockJobResponse),
          });
        }
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve([]) });
      })
    );

    render(
      <DarkVesselControlPanel
        onSceneSelect={mockOnSceneSelect}
        onDetectionsLoaded={mockOnDetectionsLoaded}
        onSelectDetection={mockOnSelectDetection}
        selectedDetection={ambiguousCoastalDetection}
      />
    );

    // Search and analyze scene
    fireEvent.click(screen.getByRole('button', { name: /Search Sentinel-1 Scenes/i }));
    await waitFor(() => screen.getByText('Sentinel-1A'));
    fireEvent.click(screen.getByText('Sentinel-1A'));
    fireEvent.click(screen.getByRole('button', { name: /Analyze Scene & Correlate AIS/i }));

    await waitFor(() => {
      expect(screen.getByTestId('operational-vessels-count')).toHaveTextContent('1 Operational Vessels');
    });

    // Coastal candidate is counted in operational count
    expect(screen.getByTestId('matched-count')).toHaveTextContent('1');

    // Nearshore boundary ambiguity is visibly displayed (not silently relabeled)
    expect(screen.getByText('COASTAL AMBIGUITY (≤10 km)')).toBeInTheDocument();
    expect(screen.getByText('AMBIGUOUS_COASTAL')).toBeInTheDocument();
    expect(screen.getByText(/4.2 km of coastline/i)).toBeInTheDocument();
  });

  it('maintains backwards compatibility with legacy GRD response lacking geographic-context fields', async () => {
    // Legacy GRD response: no surface_context, no is_operational_vessel, no geographic_context in provenance
    const legacyGRDDetection: SARDetectionItem = {
      detection_id: 'sar-det-legacy-01',
      scene_id: 'S1A_IW_GRDH_1SDV_20240901',
      acquisition_time: '2024-09-01T12:00:00Z',
      latitude: 50.5,
      longitude: 0.5,
      confidence: 0.91,
      model_version: 'yolo26s-sar-v1',
      match_status: 'AIS-MATCHED',
      candidate_mmsi: 235123456,
      distance_km: 0.4,
      time_difference_seconds: 60,
    };

    const mockLegacyJobResponse: SARJobResponse = {
      job_id: 'sar-job-legacy-grd',
      scene_id: 'S1A_IW_GRDH_1SDV_20240901',
      status: 'COMPLETE',
      progress_percent: 100,
      message: 'Analysis complete: 1 operational vessels detected',
      detections_count: 1,
      matched_count: 1,
      unmatched_count: 0,
      created_at: '2024-09-01T12:05:00Z',
      detections: [legacyGRDDetection],
    };

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (url.includes('/api/v1/sar/search')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve([
              {
                ...mockScene,
                scene_id: 'S1A_IW_GRDH_1SDV_20240901',
                processing_level: 'LEVEL1_GRD',
                modality: 'GRD',
              },
            ]),
          });
        }
        if (url.includes('/api/v1/sar/jobs/')) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve(mockLegacyJobResponse),
          });
        }
        if (url.endsWith('/api/v1/sar/jobs')) {
          return Promise.resolve({
            ok: true,
            status: 202,
            json: () => Promise.resolve(mockLegacyJobResponse),
          });
        }
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve([]) });
      })
    );

    render(
      <DarkVesselControlPanel
        onSceneSelect={mockOnSceneSelect}
        onDetectionsLoaded={mockOnDetectionsLoaded}
        onSelectDetection={mockOnSelectDetection}
        selectedDetection={legacyGRDDetection}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /Search Sentinel-1 Scenes/i }));
    await waitFor(() => screen.getByText('Sentinel-1A'));
    fireEvent.click(screen.getByText('Sentinel-1A'));
    fireEvent.click(screen.getByRole('button', { name: /Analyze Scene & Correlate AIS/i }));

    await waitFor(() => {
      expect(screen.getByTestId('operational-vessels-count')).toHaveTextContent('1 Operational Vessels');
    });

    // Counts correctly without error
    expect(screen.getByTestId('matched-count')).toHaveTextContent('1');
    expect(screen.getByTestId('unmatched-count')).toHaveTextContent('0');

    // Renders provenance cleanly with fallback
    expect(screen.getByText('OPERATIONAL VESSEL')).toBeInTheDocument();
    expect(screen.getByText('PLAUSIBLE_WATER')).toBeInTheDocument();
    expect(screen.getByText('yolo26s-sar-v1')).toBeInTheDocument();
  });
});
