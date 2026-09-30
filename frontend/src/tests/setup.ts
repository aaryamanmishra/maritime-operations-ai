import '@testing-library/jest-dom';
import { vi } from 'vitest';

vi.mock('maplibre-gl', () => {
  class MockMap {
    addControl = vi.fn();
    on = vi.fn();
    remove = vi.fn();
    addSource = vi.fn();
    addLayer = vi.fn();
    getSource = vi.fn().mockReturnValue({ setData: vi.fn() });
    isStyleLoaded = vi.fn().mockReturnValue(true);
    getBounds = vi.fn().mockReturnValue({
      getWest: () => -2.0,
      getSouth: 49.5,
      getEast: 2.5,
      getNorth: 51.5,
    });
    getCanvas = vi.fn().mockReturnValue({ style: {} });
  }

  return {
    default: {
      Map: MockMap,
      NavigationControl: vi.fn(),
      AttributionControl: vi.fn(),
    },
    Map: MockMap,
    NavigationControl: vi.fn(),
    AttributionControl: vi.fn(),
  };
});

// Mock WebSocket in Node/JSDOM
class MockWebSocket {
  static OPEN = 1;
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;

  send = vi.fn();
  close = vi.fn();
}

vi.stubGlobal('WebSocket', MockWebSocket);
