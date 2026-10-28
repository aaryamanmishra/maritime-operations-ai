import os
from pathlib import Path

fig_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\maritime-operations-ai\reports\academic_report\figures")
root_fig_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\reports\academic_report\figures")

def create_svg(filename, width, height, title, content_svg):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <style>
    .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 17px; font-weight: bold; fill: #0f172a; text-anchor: middle; }}
    .box-rect {{ fill: #f8fafc; stroke: #3b82f6; stroke-width: 2; rx: 8; }}
    .box-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: bold; fill: #1e3a8a; text-anchor: middle; }}
    .box-text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; text-anchor: middle; }}
    .edge-line {{ stroke: #2563eb; stroke-width: 2; fill: none; }}
    .edge-dash {{ stroke: #64748b; stroke-width: 1.5; stroke-dasharray: 4; fill: none; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width/2}" y="30" class="title">{title}</text>
  {content_svg}
</svg>'''
    (fig_dir / filename).write_text(svg, encoding="utf-8")
    (root_fig_dir / filename).write_text(svg, encoding="utf-8")
    print(f"Rendered vector SVG: {filename}")

# Fig 3.2: End-to-End Data Flow Sequence
c_3_2 = '''
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <!-- Lifelines -->
  <line x1="80" y1="80" x2="80" y2="420" class="edge-dash" />
  <line x1="220" y1="80" x2="220" y2="420" class="edge-dash" />
  <line x1="380" y1="80" x2="380" y2="420" class="edge-dash" />
  <line x1="540" y1="80" x2="540" y2="420" class="edge-dash" />
  <line x1="700" y1="80" x2="700" y2="420" class="edge-dash" />
  <line x1="840" y1="80" x2="840" y2="420" class="edge-dash" />

  <!-- Headers -->
  <rect x="20" y="50" width="120" height="30" fill="#e0f2fe" stroke="#0284c7" rx="4" />
  <text x="80" y="70" class="box-title">Watch Officer</text>
  <rect x="160" y="50" width="120" height="30" fill="#f1f5f9" stroke="#475569" rx="4" />
  <text x="220" y="70" class="box-title">React Frontend</text>
  <rect x="320" y="50" width="120" height="30" fill="#f8fafc" stroke="#2563eb" rx="4" />
  <text x="380" y="70" class="box-title">FastAPI Backend</text>
  <rect x="480" y="50" width="120" height="30" fill="#eff6ff" stroke="#1d4ed8" rx="4" />
  <text x="540" y="70" class="box-title">Copernicus CDSE</text>
  <rect x="640" y="50" width="120" height="30" fill="#fef2f2" stroke="#dc2626" rx="4" />
  <text x="700" y="70" class="box-title">YOLO Detector</text>
  <rect x="780" y="50" width="120" height="30" fill="#f0fdf4" stroke="#16a34a" rx="4" />
  <text x="840" y="70" class="box-title">PostGIS &amp; AIS</text>

  <!-- Messages -->
  <line x1="80" y1="110" x2="220" y2="110" class="edge-line" marker-end="url(#arr2)" />
  <text x="150" y="105" class="box-text">1. Select BBox &amp; Dates</text>

  <line x1="220" y1="140" x2="380" y2="140" class="edge-line" marker-end="url(#arr2)" />
  <text x="300" y="135" class="box-text">2. POST /api/v1/sar/search</text>

  <line x1="380" y1="170" x2="540" y2="170" class="edge-line" marker-end="url(#arr2)" />
  <text x="460" y="165" class="box-text">3. STAC Scene Query</text>

  <line x1="540" y1="200" x2="220" y2="200" class="edge-line" stroke-dasharray="4" marker-end="url(#arr2)" />
  <text x="380" y="195" class="box-text">4. Scene Footprints to UI</text>

  <line x1="80" y1="230" x2="220" y2="230" class="edge-line" marker-end="url(#arr2)" />
  <text x="150" y="225" class="box-text">5. Click 'Analyze Scene'</text>

  <line x1="220" y1="260" x2="380" y2="260" class="edge-line" marker-end="url(#arr2)" />
  <text x="300" y="255" class="box-text">6. POST /api/v1/sar/jobs (202 Accepted)</text>

  <line x1="380" y1="290" x2="540" y2="290" class="edge-line" marker-end="url(#arr2)" />
  <text x="460" y="285" class="box-text">7. Download GeoTIFF Bands</text>

  <line x1="380" y1="320" x2="700" y2="320" class="edge-line" marker-end="url(#arr2)" />
  <text x="540" y="315" class="box-text">8. Tiled Inference (640x640)</text>

  <line x1="700" y1="350" x2="380" y2="350" class="edge-line" marker-end="url(#arr2)" />
  <text x="540" y="345" class="box-text">9. Detections &amp; GCP Georeference</text>

  <line x1="380" y1="380" x2="840" y2="380" class="edge-line" marker-end="url(#arr2)" />
  <text x="610" y="375" class="box-text">10. Spatiotemporal AIS Match (+-15m, 3km)</text>

  <line x1="380" y1="410" x2="220" y2="410" class="edge-line" stroke-dasharray="4" marker-end="url(#arr2)" />
  <text x="300" y="405" class="box-text">11. Render Dark-Vessel Overlays</text>
'''
create_svg("fig3_2_end_to_end_data_flow.svg", 940, 460, "Fig 3.2: Sentinel-1 and AIS Dark-Vessel Analysis End-to-End Data Flow", c_3_2)

# Fig 3.3: SAR Acquisition & Preprocessing Flowchart
c_3_3 = '''
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect x="250" y="50" width="300" height="40" class="box-rect" />
  <text x="400" y="75" class="box-title">Sentinel-1 Scene Selected from CDSE Catalog</text>

  <line x1="400" y1="90" x2="400" y2="120" class="edge-line" marker-end="url(#arr3)" />

  <polygon points="400,120 520,155 400,190 280,155" fill="#fef3c7" stroke="#d97706" stroke-width="1.5" />
  <text x="400" y="152" font-family="sans-serif" font-weight="bold" font-size="11px" fill="#92400e" text-anchor="middle">Scene in Local Cache?</text>

  <line x1="520" y1="155" x2="620" y2="155" class="edge-line" />
  <line x1="620" y1="155" x2="620" y2="210" class="edge-line" marker-end="url(#arr3)" />
  <text x="560" y="145" class="box-text">No</text>
  <rect x="520" y="210" width="200" height="40" class="box-rect" />
  <text x="620" y="235" class="box-title">Stream OData Zip/SAFE Archive</text>

  <line x1="280" y1="155" x2="180" y2="155" class="edge-line" />
  <line x1="180" y1="155" x2="180" y2="210" class="edge-line" marker-end="url(#arr3)" />
  <text x="240" y="145" class="box-text">Yes</text>
  <rect x="80" y="210" width="200" height="40" class="box-rect" />
  <text x="180" y="235" class="box-title">Load GeoTIFF from Disk</text>

  <line x1="180" y1="250" x2="350" y2="290" class="edge-line" marker-end="url(#arr3)" />
  <line x1="620" y1="250" x2="450" y2="290" class="edge-line" marker-end="url(#arr3)" />

  <rect x="250" y="290" width="300" height="40" class="box-rect" />
  <text x="400" y="315" class="box-title">Extract VV and VH Measurement Bands</text>

  <line x1="400" y1="330" x2="400" y2="360" class="edge-line" marker-end="url(#arr3)" />

  <rect x="220" y="360" width="360" height="45" class="box-rect" />
  <text x="400" y="380" class="box-title">Percentile Clip Normalization (2.0% - 99.5%)</text>
  <text x="400" y="395" class="box-text">Eliminates extreme speckle outliers, maps to 0-255 uint8</text>

  <line x1="400" y1="405" x2="400" y2="435" class="edge-line" marker-end="url(#arr3)" />

  <rect x="220" y="435" width="360" height="45" class="box-rect" />
  <text x="400" y="455" class="box-title">Dual-Polarization Fusion Stack</text>
  <text x="400" y="470" class="box-text">Channel 0 = VV, Channel 1 = VH, Channel 2 = (VV+VH)/2</text>

  <line x1="400" y1="480" x2="400" y2="510" class="edge-line" marker-end="url(#arr3)" />

  <rect x="220" y="510" width="360" height="40" fill="#eff6ff" stroke="#2563eb" stroke-width="2" rx="6" />
  <text x="400" y="535" class="box-title">Native-Resolution 640x640 Sliding-Window Tiler</text>
'''
create_svg("fig3_3_sar_acquisition_preprocessing_flowchart.svg", 800, 580, "Fig 3.3: Sentinel-1 SAR Acquisition and Preprocessing Flowchart", c_3_3)

# Fig 3.4: Native Resolution Chip Extraction
c_3_4 = '''
  <defs>
    <marker id="arr4" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect x="50" y="70" width="320" height="260" fill="#f8fafc" stroke="#334155" stroke-width="2" rx="6" />
  <text x="210" y="95" class="box-title">Full Scene GeoTIFF (25,000 x 16,500 px)</text>
  
  <!-- Grid representation inside full scene -->
  <rect x="80" y="110" width="80" height="80" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5" />
  <text x="120" y="155" class="box-text">Chip (0,0)</text>

  <rect x="150" y="110" width="80" height="80" fill="#bfdbfe" stroke="#1d4ed8" stroke-width="1.5" stroke-dasharray="3" />
  <text x="190" y="155" class="box-text">Chip (1,0)</text>

  <rect x="80" y="180" width="80" height="80" fill="#bfdbfe" stroke="#1d4ed8" stroke-width="1.5" stroke-dasharray="3" />
  <text x="120" y="225" class="box-text">Chip (0,1)</text>

  <text x="210" y="285" class="box-text">Chip Size = 640 x 640 px</text>
  <text x="210" y="305" class="box-text">Stride = 576 px | Overlap = 64 px (10%)</text>

  <line x1="370" y1="200" x2="450" y2="200" class="edge-line" marker-end="url(#arr4)" />
  <text x="410" y="190" class="box-text">Extract</text>

  <rect x="470" y="80" width="260" height="240" fill="#eff6ff" stroke="#2563eb" stroke-width="2" rx="6" />
  <text x="600" y="110" class="box-title">Extracted Radar Chip (640 x 640)</text>
  <rect x="520" y="130" width="160" height="130" fill="#1e293b" rx="4" />
  <circle cx="580" cy="180" r="4" fill="#ffffff" />
  <rect x="572" y="172" width="16" height="16" fill="none" stroke="#22c55e" stroke-width="1.5" />
  <text x="600" y="280" class="box-text">Preserves native 10m x 10m spatial pixel size</text>
  <text x="600" y="295" class="box-text">Local-to-Global offset recorded (x_off, y_off)</text>
'''
create_svg("fig3_4_native_resolution_chip_extraction.svg", 780, 360, "Fig 3.4: Sliding-Window Native-Resolution Radar Chip Extraction", c_3_4)

# Fig 3.5: Ship Detection & Inference Pipeline
c_3_5 = '''
  <defs>
    <marker id="arr5" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect x="50" y="60" width="180" height="40" class="box-rect" />
  <text x="140" y="85" class="box-title">640x640 Radar Chip</text>

  <line x1="230" y1="80" x2="280" y2="80" class="edge-line" marker-end="url(#arr5)" />

  <rect x="280" y="55" width="220" height="50" fill="#eff6ff" stroke="#2563eb" stroke-width="2" rx="6" />
  <text x="390" y="77" class="box-title">Modified CSPDarknet</text>
  <text x="390" y="93" class="box-text">Multi-scale feature extraction</text>

  <line x1="500" y1="80" x2="550" y2="80" class="edge-line" marker-end="url(#arr5)" />

  <rect x="550" y="55" width="220" height="50" fill="#f8fafc" stroke="#475569" stroke-width="2" rx="6" />
  <text x="660" y="77" class="box-title">PANet Neck + SPPF</text>
  <text x="660" y="93" class="box-text">Top-down &amp; bottom-up fusion</text>

  <line x1="660" y1="105" x2="660" y2="150" class="edge-line" marker-end="url(#arr5)" />

  <rect x="550" y="150" width="220" height="50" fill="#fdf4ff" stroke="#a855f7" stroke-width="2" rx="6" />
  <text x="660" y="172" class="box-title">Decoupled Anchor-Free Head</text>
  <text x="660" y="188" class="box-text">Class: Vessel | Reg: xc, yc, w, h</text>

  <line x1="550" y1="175" x2="480" y2="175" class="edge-line" marker-end="url(#arr5)" />

  <rect x="280" y="150" width="200" height="50" class="box-rect" />
  <text x="380" y="172" class="box-title">Confidence Gate (&gt;= 0.25)</text>
  <text x="380" y="188" class="box-text">Suppresses sea surface clutter</text>

  <line x1="280" y1="175" x2="230" y2="175" class="edge-line" marker-end="url(#arr5)" />

  <rect x="50" y="150" width="180" height="50" class="box-rect" />
  <text x="140" y="172" class="box-title">Intra-Tile NMS (IoU 0.35)</text>
  <text x="140" y="188" class="box-text">Local peak consolidation</text>

  <line x1="140" y1="200" x2="140" y2="245" class="edge-line" marker-end="url(#arr5)" />

  <rect x="30" y="245" width="220" height="50" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" rx="6" />
  <text x="140" y="267" class="box-title">Cross-Tile Deduplication</text>
  <text x="140" y="283" class="box-text">IoU &gt;= 0.30 in overlap band</text>

  <line x1="250" y1="270" x2="320" y2="270" class="edge-line" marker-end="url(#arr5)" />

  <rect x="320" y="245" width="240" height="50" fill="#dcfce7" stroke="#15803d" stroke-width="2" rx="6" />
  <text x="440" y="267" class="box-title">Operational SAR Detection</text>
  <text x="440" y="283" class="box-text">Single primary contact with provenance</text>
'''
create_svg("fig3_5_ship_detection_inference_pipeline.svg", 800, 330, "Fig 3.5: YOLO Anchor-Free Ship Detection and Deduplication Pipeline", c_3_5)

# Fig 3.6: Pixel-to-Geographic Coordinate Transformation Flowchart
c_3_6 = '''
  <defs>
    <marker id="arr6" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect x="250" y="50" width="300" height="40" class="box-rect" />
  <text x="400" y="75" class="box-title">BBox Center Pixel (X = x + x_off, Y = y + y_off)</text>

  <line x1="400" y1="90" x2="400" y2="125" class="edge-line" marker-end="url(#arr6)" />

  <rect x="220" y="125" width="360" height="45" class="box-rect" />
  <text x="400" y="145" class="box-title">Bisection Line Search in GCP Grid</text>
  <text x="400" y="160" class="box-text">Find bounding lines [L0, L1] enclosing target line Y</text>

  <line x1="400" y1="170" x2="400" y2="205" class="edge-line" marker-end="url(#arr6)" />

  <rect x="220" y="205" width="360" height="45" class="box-rect" />
  <text x="400" y="225" class="box-title">Bisection Pixel Search along Lines</text>
  <text x="400" y="240" class="box-text">Find bounding columns [P0, P1] enclosing target pixel X</text>

  <line x1="400" y1="250" x2="400" y2="285" class="edge-line" marker-end="url(#arr6)" />

  <rect x="200" y="285" width="400" height="50" fill="#eff6ff" stroke="#2563eb" stroke-width="2" rx="6" />
  <text x="400" y="307" class="box-title">Piecewise Bilinear Geodesic Interpolation</text>
  <text x="400" y="323" class="box-text">u = (X - P0)/(P1 - P0), v = (Y - L0)/(L1 - L0) -&gt; Lon, Lat</text>

  <line x1="400" y1="335" x2="400" y2="370" class="edge-line" marker-end="url(#arr6)" />

  <rect x="220" y="370" width="360" height="45" class="box-rect" />
  <text x="400" y="390" class="box-title">Antimeridian Normalization &amp; Invariants</text>
  <text x="400" y="405" class="box-text">Lon in [-180, 180), Lat in [-90, 90], finite checks</text>

  <line x1="400" y1="415" x2="400" y2="450" class="edge-line" marker-end="url(#arr6)" />

  <rect x="220" y="450" width="360" height="40" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" rx="6" />
  <text x="400" y="475" class="box-title">Validated GeoPoint (WGS84 EPSG:4326)</text>
'''
create_svg("fig3_6_pixel_to_geo_coordinate_transformation_flowchart.svg", 800, 520, "Fig 3.6: GCP-Based Piecewise Bilinear Georeferencing Flowchart", c_3_6)

# Fig 3.7: AIS Ingestion & Storage Architecture
c_3_7 = '''
  <defs>
    <marker id="arr7" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect x="30" y="70" width="200" height="200" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" rx="6" />
  <text x="130" y="95" class="box-title">AIS Ingestion Worker</text>
  <rect x="45" y="115" width="170" height="35" class="box-rect" />
  <text x="130" y="137" class="box-text">AIVDM Message Parser</text>
  <rect x="45" y="160" width="170" height="35" class="box-rect" />
  <text x="130" y="182" class="box-text">BBox Filter (49.5-51.5N)</text>
  <rect x="45" y="205" width="170" height="35" class="box-rect" />
  <text x="130" y="227" class="box-text">Kinematic Sanity Filter</text>

  <line x1="230" y1="130" x2="310" y2="130" class="edge-line" marker-end="url(#arr7)" />
  <line x1="230" y1="210" x2="310" y2="210" class="edge-line" marker-end="url(#arr7)" />

  <rect x="310" y="70" width="220" height="110" fill="#fef2f2" stroke="#dc2626" stroke-width="2" rx="6" />
  <text x="420" y="95" class="box-title">Redis 7 Ephemeral Cache</text>
  <text x="420" y="120" class="box-text">GEOADD live_vessel_positions</text>
  <text x="420" y="140" class="box-text">Pub/Sub: vessel_traffic_stream</text>
  <text x="420" y="160" class="box-text">TTL Expiry = 3600 seconds</text>

  <rect x="310" y="200" width="220" height="130" fill="#eff6ff" stroke="#2563eb" stroke-width="2" rx="6" />
  <text x="420" y="225" class="box-title">PostGIS Spatial Database</text>
  <text x="420" y="250" class="box-text">current_vessel_state (MMSI PK)</text>
  <text x="420" y="270" class="box-text">vessel_position_history (Partitioned)</text>
  <text x="420" y="290" class="box-text">GIST Spatial Index on geom</text>
  <text x="420" y="310" class="box-text">B-Tree on (mmsi, timestamp)</text>
'''
create_svg("fig3_7_ais_ingestion_storage_architecture.svg", 580, 360, "Fig 3.7: AIS Ingestion, In-Memory Caching, and PostGIS Storage Architecture", c_3_7)

# Fig 3.8: SAR/AIS Temporal-Spatial Matching Flowchart
c_3_8 = '''
  <defs>
    <marker id="arr8" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect x="220" y="50" width="360" height="40" class="box-rect" />
  <text x="400" y="75" class="box-title">SAR Detection Contact (Lat, Lon, T_sat)</text>

  <line x1="400" y1="90" x2="400" y2="125" class="edge-line" marker-end="url(#arr8)" />

  <rect x="200" y="125" width="400" height="45" class="box-rect" />
  <text x="400" y="145" class="box-title">ST_DWithin PostGIS Candidate Query</text>
  <text x="400" y="160" class="box-text">Window: [T_sat - 15m, T_sat + 15m] | Gate: 4500m</text>

  <line x1="400" y1="170" x2="400" y2="205" class="edge-line" marker-end="url(#arr8)" />

  <rect x="200" y="205" width="400" height="50" fill="#eff6ff" stroke="#2563eb" stroke-width="2" rx="6" />
  <text x="400" y="227" class="box-title">Kinematic Dead Reckoning Projection</text>
  <text x="400" y="243" class="box-text">est_pos = AIS_pos + SOG * dt * [cos COG, sin COG]</text>

  <line x1="400" y1="255" x2="400" y2="290" class="edge-line" marker-end="url(#arr8)" />

  <rect x="180" y="290" width="440" height="50" class="box-rect" />
  <text x="400" y="312" class="box-title">Positional Uncertainty Budget Calculation</text>
  <text x="400" y="328" class="box-text">sigma_total = sqrt(sigma_sar^2 + sigma_ais^2 + sigma_motion^2)</text>

  <line x1="400" y1="340" x2="400" y2="375" class="edge-line" marker-end="url(#arr8)" />

  <rect x="200" y="375" width="400" height="45" class="box-rect" />
  <text x="400" y="395" class="box-title">Z-Score &amp; Gaussian Likelihood Score</text>
  <text x="400" y="410" class="box-text">z = dist_m / sigma_total | score = exp(-0.5 * z^2)</text>

  <line x1="400" y1="420" x2="400" y2="455" class="edge-line" marker-end="url(#arr8)" />

  <rect x="220" y="455" width="360" height="40" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" rx="6" />
  <text x="400" y="480" class="box-title">Decision Gate Evaluation &amp; Match Record</text>
'''
create_svg("fig3_8_sar_ais_temporal_spatial_matching_flowchart.svg", 800, 520, "Fig 3.8: Kinematic Uncertainty-Aware Spatiotemporal Correlation Flowchart", c_3_8)

# Fig 3.11: UML Use Case Diagram
c_3_11 = '''
  <defs>
    <marker id="arr11" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <!-- Actors -->
  <rect x="40" y="80" width="140" height="50" fill="#eff6ff" stroke="#2563eb" rx="6" />
  <text x="110" y="105" class="box-title">Maritime Watch</text>
  <text x="110" y="120" class="box-title">Officer</text>

  <rect x="40" y="190" width="140" height="50" fill="#eff6ff" stroke="#2563eb" rx="6" />
  <text x="110" y="215" class="box-title">Intelligence</text>
  <text x="110" y="230" class="box-title">Analyst</text>

  <rect x="40" y="300" width="140" height="50" fill="#f1f5f9" stroke="#64748b" rx="6" />
  <text x="110" y="325" class="box-title">System</text>
  <text x="110" y="340" class="box-title">Administrator</text>

  <!-- Use cases -->
  <ellipse cx="380" cy="80" rx="130" ry="25" fill="#f8fafc" stroke="#0284c7" stroke-width="1.5" />
  <text x="380" y="85" class="box-text">UC-01: Monitor Live Traffic</text>

  <ellipse cx="380" cy="140" rx="130" ry="25" fill="#f8fafc" stroke="#0284c7" stroke-width="1.5" />
  <text x="380" y="145" class="box-text">UC-02: Query Satellite Swaths</text>

  <ellipse cx="380" cy="200" rx="130" ry="25" fill="#f8fafc" stroke="#0284c7" stroke-width="1.5" />
  <text x="380" y="205" class="box-text">UC-03: Execute Ship Detection</text>

  <ellipse cx="380" cy="260" rx="130" ry="25" fill="#f8fafc" stroke="#0284c7" stroke-width="1.5" />
  <text x="380" y="265" class="box-text">UC-04: Classify Dark Vessels</text>

  <ellipse cx="380" cy="320" rx="130" ry="25" fill="#f8fafc" stroke="#0284c7" stroke-width="1.5" />
  <text x="380" y="325" class="box-text">UC-05: Inspect Provenance</text>

  <ellipse cx="380" cy="380" rx="130" ry="25" fill="#f8fafc" stroke="#0284c7" stroke-width="1.5" />
  <text x="380" y="385" class="box-text">UC-06: Plan Weather Route</text>

  <!-- Connectors -->
  <line x1="180" y1="105" x2="250" y2="80" class="edge-line" />
  <line x1="180" y1="105" x2="250" y2="140" class="edge-line" />
  <line x1="180" y1="105" x2="250" y2="200" class="edge-line" />
  <line x1="180" y1="105" x2="250" y2="380" class="edge-line" />

  <line x1="180" y1="215" x2="250" y2="200" class="edge-line" />
  <line x1="180" y1="215" x2="250" y2="260" class="edge-line" />
  <line x1="180" y1="215" x2="250" y2="320" class="edge-line" />

  <line x1="180" y1="325" x2="250" y2="380" class="edge-line" />
'''
create_svg("fig3_11_uml_use_case_diagram.svg", 580, 430, "Fig 3.11: Maritime Operations AI UML Use-Case Diagram", c_3_11)

# Fig 3.12: Backend & Frontend Interaction Diagram
c_3_12 = '''
  <defs>
    <marker id="arr12" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <!-- Lifelines -->
  <line x1="100" y1="80" x2="100" y2="360" class="edge-dash" />
  <line x1="300" y1="80" x2="300" y2="360" class="edge-dash" />
  <line x1="500" y1="80" x2="500" y2="360" class="edge-dash" />
  <line x1="700" y1="80" x2="700" y2="360" class="edge-dash" />

  <!-- Headers -->
  <rect x="30" y="50" width="140" height="30" fill="#f1f5f9" stroke="#334155" rx="4" />
  <text x="100" y="70" class="box-title">MaritimeMap.tsx</text>

  <rect x="220" y="50" width="160" height="30" fill="#eff6ff" stroke="#2563eb" rx="4" />
  <text x="300" y="70" class="box-title">DarkVesselControlPanel</text>

  <rect x="430" y="50" width="140" height="30" fill="#fdf4ff" stroke="#a855f7" rx="4" />
  <text x="500" y="70" class="box-title">sarApi.ts Client</text>

  <rect x="630" y="50" width="140" height="30" fill="#f0fdf4" stroke="#16a34a" rx="4" />
  <text x="700" y="70" class="box-title">FastAPI Backend</text>

  <!-- Sequence lines -->
  <line x1="100" y1="110" x2="300" y2="110" class="edge-line" marker-end="url(#arr12)" />
  <text x="200" y="105" class="box-text">1. BBox Selected on Map</text>

  <line x1="300" y1="140" x2="500" y2="140" class="edge-line" marker-end="url(#arr12)" />
  <text x="400" y="135" class="box-text">2. searchSARScenes()</text>

  <line x1="500" y1="170" x2="700" y2="170" class="edge-line" marker-end="url(#arr12)" />
  <text x="600" y="165" class="box-text">3. POST /api/v1/sar/search</text>

  <line x1="700" y1="200" x2="300" y2="200" class="edge-line" stroke-dasharray="4" marker-end="url(#arr12)" />
  <text x="500" y="195" class="box-text">4. Display Available Scenes</text>

  <line x1="300" y1="230" x2="700" y2="230" class="edge-line" marker-end="url(#arr12)" />
  <text x="500" y="225" class="box-text">5. POST /api/v1/sar/jobs (start analysis)</text>

  <line x1="500" y1="260" x2="700" y2="260" class="edge-line" marker-end="url(#arr12)" />
  <text x="600" y="255" class="box-text">6. Poll GET /jobs/{id}</text>

  <line x1="700" y1="290" x2="500" y2="290" class="edge-line" stroke-dasharray="4" marker-end="url(#arr12)" />
  <text x="600" y="285" class="box-text">7. Job Status: COMPLETED</text>

  <line x1="500" y1="320" x2="700" y2="320" class="edge-line" marker-end="url(#arr12)" />
  <text x="600" y="315" class="box-text">8. GET /scenes/{id}/detections</text>

  <line x1="700" y1="345" x2="100" y2="345" class="edge-line" stroke-dasharray="4" marker-end="url(#arr12)" />
  <text x="400" y="340" class="box-text">9. Render Overlays &amp; Provenance on Map</text>
'''
create_svg("fig3_12_backend_frontend_interaction.svg", 800, 390, "Fig 3.12: Tactical Control Panel and REST API Interaction Sequence", c_3_12)

print("All 12 vector SVGs rendered successfully.")
