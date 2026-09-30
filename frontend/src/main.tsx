import React from 'react';
import ReactDOM from 'react-dom/client';
import * as maplibregl from 'maplibre-gl';
import App from './App';
import './index.css';

// Fix MapLibre worker URL for Vite production builds.
// When Vite bundles maplibre-gl.mjs, import.meta.url inside that bundle points
// to the app chunk URL (e.g. /assets/index-HASH.js), not the maplibre package.
// As a result, the auto-derived worker URL (/assets/maplibre-gl-worker.mjs)
// returns HTTP 404 and tiles never parse. We must set the correct URL explicitly.
// The worker file is served from /public/maplibre-gl-worker.mjs → /maplibre-gl-worker.mjs.
maplibregl.setWorkerUrl('/maplibre-gl-worker.mjs');

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
