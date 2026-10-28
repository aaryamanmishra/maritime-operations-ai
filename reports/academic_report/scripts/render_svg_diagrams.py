import os
from pathlib import Path

fig_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\maritime-operations-ai\reports\academic_report\figures")
root_fig_dir = Path(r"C:\Users\Aayush Barsainya\Downloads\DL_CourseProject\reports\academic_report\figures")

def create_svg(filename, width, height, title, content_svg):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <style>
    .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #0f172a; text-anchor: middle; }}
    .subtitle {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }}
    .box-rect {{ fill: #f8fafc; stroke: #3b82f6; stroke-width: 2; rx: 8; }}
    .box-header {{ fill: #eff6ff; stroke: #3b82f6; stroke-width: 1.5; rx: 6; }}
    .box-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: bold; fill: #1e3a8a; text-anchor: middle; }}
    .box-text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; text-anchor: middle; }}
    .edge-line {{ stroke: #2563eb; stroke-width: 2; fill: none; }}
    .edge-dash {{ stroke: #64748b; stroke-width: 1.5; stroke-dasharray: 4; fill: none; }}
    .edge-label {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 10px; fill: #475569; text-anchor: middle; background: white; }}
    .status-dark {{ fill: #fee2e2; stroke: #dc2626; stroke-width: 2; rx: 6; }}
    .status-matched {{ fill: #dcfce7; stroke: #16a34a; stroke-width: 2; rx: 6; }}
    .status-review {{ fill: #fef3c7; stroke: #d97706; stroke-width: 2; rx: 6; }}
    .status-text-dark {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: bold; fill: #991b1b; text-anchor: middle; }}
    .status-text-matched {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: bold; fill: #166534; text-anchor: middle; }}
    .status-text-review {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: bold; fill: #92400e; text-anchor: middle; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width/2}" y="30" class="title">{title}</text>
  {content_svg}
</svg>'''
    (fig_dir / filename).write_text(svg, encoding="utf-8")
    (root_fig_dir / filename).write_text(svg, encoding="utf-8")
    print(f"Rendered vector SVG: {filename}")

# Fig 3.1: Overall System Architecture
c_3_1 = '''
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <!-- Outer groups -->
  <rect x="30" y="60" width="220" height="150" fill="#f0f9ff" stroke="#0284c7" stroke-width="1.5" rx="8" />
  <text x="140" y="85" class="box-title">Copernicus Sentinel-1 SAR</text>
  <rect x="45" y="105" width="190" height="40" class="box-rect" />
  <text x="140" y="125" class="box-text">Sentinel-1A / 1C Radar</text>
  <text x="140" y="138" class="box-text">C-Band IW 5.405 GHz</text>
  <rect x="45" y="155" width="190" height="40" class="box-rect" />
  <text x="140" y="175" class="box-text">Copernicus CDSE STAC</text>
  <text x="140" y="188" class="box-text">OData Level-1 GRD / SLC</text>

  <rect x="30" y="240" width="220" height="150" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5" rx="8" />
  <text x="140" y="265" class="box-title">AIS Broadcast Ingestion</text>
  <rect x="45" y="285" width="190" height="40" class="box-rect" />
  <text x="140" y="305" class="box-text">AISStream.io WebSocket</text>
  <text x="140" y="318" class="box-text">AIVDM/AIVDO JSON Stream</text>
  <rect x="45" y="335" width="190" height="40" class="box-rect" />
  <text x="140" y="355" class="box-text">AIS Ingestion Worker</text>
  <text x="140" y="368" class="box-text">AsyncIO Parser &amp; BBox Filter</text>

  <rect x="290" y="60" width="420" height="330" fill="#f8fafc" stroke="#334155" stroke-width="2" rx="8" />
  <text x="500" y="85" class="box-title">Maritime Operations AI Platform Core (FastAPI / PyTorch)</text>
  
  <rect x="310" y="105" width="180" height="50" class="box-rect" />
  <text x="400" y="125" class="box-title">Acquisition Adapter</text>
  <text x="400" y="142" class="box-text">CDSE OData Downloader</text>

  <rect x="510" y="105" width="180" height="50" class="box-rect" />
  <text x="600" y="125" class="box-title">Preprocessing Engine</text>
  <text x="600" y="142" class="box-text">Percentile Norm &amp; 640 Tiler</text>

  <rect x="310" y="175" width="180" height="50" class="box-rect" />
  <text x="400" y="195" class="box-title">YOLO Vessel Detector</text>
  <text x="400" y="212" class="box-text">YOLO11n / YOLOv8s Head</text>

  <rect x="510" y="175" width="180" height="50" class="box-rect" />
  <text x="600" y="195" class="box-title">GCP Georeferencer</text>
  <text x="600" y="212" class="box-text">Piecewise Bilinear Grid</text>

  <rect x="310" y="245" width="180" height="50" class="box-rect" />
  <text x="400" y="265" class="box-title">Correlation Engine</text>
  <text x="400" y="282" class="box-text">Kinematic Uncertainty Gates</text>

  <rect x="510" y="245" width="180" height="50" class="box-rect" />
  <text x="600" y="265" class="box-title">Weather Routing Engine</text>
  <text x="600" y="282" class="box-text">TFT SOG Loss &amp; Kwon Model</text>

  <rect x="310" y="315" width="380" height="55" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5" rx="6" />
  <text x="500" y="337" class="box-title">Geospatial Persistence Layer</text>
  <text x="500" y="355" class="box-text">PostgreSQL 16 + PostGIS 3.4 (EPSG:4326) | Redis 7 Geospatial Cache</text>

  <rect x="750" y="60" width="220" height="330" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.5" rx="8" />
  <text x="860" y="85" class="box-title">Tactical Command UI</text>
  <rect x="765" y="110" width="190" height="50" class="box-rect" />
  <text x="860" y="130" class="box-text">MapLibre GL Vector Map</text>
  <text x="860" y="145" class="box-text">High-Contrast SAR Overlays</text>
  <rect x="765" y="180" width="190" height="50" class="box-rect" />
  <text x="860" y="200" class="box-text">Dark-Vessel Tactical Panel</text>
  <text x="860" y="215" class="box-text">Target Dossier &amp; Provenance</text>
  <rect x="765" y="250" width="190" height="50" class="box-rect" />
  <text x="860" y="270" class="box-text">Real-Time Traffic Monitor</text>
  <text x="860" y="285" class="box-text">MMSI Track Interpolation</text>
  <rect x="765" y="320" width="190" height="50" class="box-rect" />
  <text x="860" y="340" class="box-text">Weather Voyage Panel</text>
  <text x="860" y="355" class="box-text">Fuel &amp; Wave Degradation</text>

  <!-- Connectors -->
  <line x1="235" y1="175" x2="310" y2="130" class="edge-line" marker-end="url(#arrow)" />
  <line x1="235" y1="355" x2="310" y2="340" class="edge-line" marker-end="url(#arrow)" />
  <line x1="710" y1="200" x2="750" y2="200" class="edge-line" marker-end="url(#arrow)" />
'''
create_svg("fig3_1_overall_system_architecture.svg", 1000, 420, "Fig 3.1: Maritime Operations AI Overall System Architecture", c_3_1)

# Fig 3.10: Database ER Diagram
c_3_10 = '''
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <!-- Tables -->
  <rect x="30" y="70" width="220" height="240" fill="#ffffff" stroke="#0284c7" stroke-width="2" rx="6" />
  <rect x="30" y="70" width="220" height="30" fill="#0284c7" />
  <text x="140" y="90" fill="#ffffff" font-family="sans-serif" font-weight="bold" font-size="12px" text-anchor="middle">sar_scene</text>
  <text x="40" y="120" font-family="monospace" font-size="10px">PK scene_id VARCHAR(128)</text>
  <text x="40" y="140" font-family="monospace" font-size="10px">acquisition_time TIMESTAMPTZ</text>
  <text x="40" y="160" font-family="monospace" font-size="10px">platform VARCHAR(64)</text>
  <text x="40" y="180" font-family="monospace" font-size="10px">orbit_pass VARCHAR(32)</text>
  <text x="40" y="200" font-family="monospace" font-size="10px">footprint GEOMETRY(POLY,4326)</text>
  <text x="40" y="220" font-family="monospace" font-size="10px">processing_level VARCHAR(32)</text>
  <text x="40" y="240" font-family="monospace" font-size="10px">polarization VARCHAR(32)</text>
  <text x="40" y="260" font-family="monospace" font-size="10px">scene_metadata JSONB</text>
  <text x="40" y="280" font-family="monospace" font-size="10px">created_at TIMESTAMPTZ</text>

  <rect x="310" y="70" width="220" height="260" fill="#ffffff" stroke="#2563eb" stroke-width="2" rx="6" />
  <rect x="310" y="70" width="220" height="30" fill="#2563eb" />
  <text x="420" y="90" fill="#ffffff" font-family="sans-serif" font-weight="bold" font-size="12px" text-anchor="middle">sar_detection</text>
  <text x="320" y="120" font-family="monospace" font-size="10px">PK id BIGINT (AUTO)</text>
  <text x="320" y="140" font-family="monospace" font-size="10px">UK detection_id VARCHAR(64)</text>
  <text x="320" y="160" font-family="monospace" font-size="10px">FK scene_id VARCHAR(128)</text>
  <text x="320" y="180" font-family="monospace" font-size="10px">geom GEOMETRY(POINT,4326)</text>
  <text x="320" y="200" font-family="monospace" font-size="10px">latitude NUMERIC(9,6)</text>
  <text x="320" y="220" font-family="monospace" font-size="10px">longitude NUMERIC(9,6)</text>
  <text x="320" y="240" font-family="monospace" font-size="10px">pixel_bbox JSONB</text>
  <text x="320" y="260" font-family="monospace" font-size="10px">confidence NUMERIC(4,3)</text>
  <text x="320" y="280" font-family="monospace" font-size="10px">model_version VARCHAR(64)</text>
  <text x="320" y="300" font-family="monospace" font-size="10px">length_m NUMERIC(6,2)</text>

  <rect x="590" y="70" width="230" height="240" fill="#ffffff" stroke="#7c3aed" stroke-width="2" rx="6" />
  <rect x="590" y="70" width="230" height="30" fill="#7c3aed" />
  <text x="705" y="90" fill="#ffffff" font-family="sans-serif" font-weight="bold" font-size="12px" text-anchor="middle">ais_sar_correlation</text>
  <text x="600" y="120" font-family="monospace" font-size="10px">PK id BIGINT (AUTO)</text>
  <text x="600" y="140" font-family="monospace" font-size="10px">FK sar_detection_id BIGINT</text>
  <text x="600" y="160" font-family="monospace" font-size="10px">FK candidate_mmsi BIGINT</text>
  <text x="600" y="180" font-family="monospace" font-size="10px">time_difference_sec NUMERIC(8,2)</text>
  <text x="600" y="200" font-family="monospace" font-size="10px">distance_km NUMERIC(8,3)</text>
  <text x="600" y="220" font-family="monospace" font-size="10px">match_status VARCHAR(32)</text>
  <text x="600" y="240" font-family="monospace" font-size="10px">matching_method VARCHAR(64)</text>
  <text x="600" y="260" font-family="monospace" font-size="10px">provenance JSONB</text>

  <rect x="310" y="370" width="220" height="200" fill="#ffffff" stroke="#059669" stroke-width="2" rx="6" />
  <rect x="310" y="370" width="220" height="30" fill="#059669" />
  <text x="420" y="390" fill="#ffffff" font-family="sans-serif" font-weight="bold" font-size="12px" text-anchor="middle">vessel_identity</text>
  <text x="320" y="420" font-family="monospace" font-size="10px">PK mmsi BIGINT</text>
  <text x="320" y="440" font-family="monospace" font-size="10px">imo INTEGER</text>
  <text x="320" y="460" font-family="monospace" font-size="10px">name VARCHAR(128)</text>
  <text x="320" y="480" font-family="monospace" font-size="10px">callsign VARCHAR(32)</text>
  <text x="320" y="500" font-family="monospace" font-size="10px">ship_type INTEGER</text>
  <text x="320" y="520" font-family="monospace" font-size="10px">length NUMERIC(8,2)</text>
  <text x="320" y="540" font-family="monospace" font-size="10px">beam NUMERIC(8,2)</text>

  <rect x="590" y="370" width="230" height="200" fill="#ffffff" stroke="#0d9488" stroke-width="2" rx="6" />
  <rect x="590" y="370" width="230" height="30" fill="#0d9488" />
  <text x="705" y="390" fill="#ffffff" font-family="sans-serif" font-weight="bold" font-size="12px" text-anchor="middle">current_vessel_state</text>
  <text x="600" y="420" font-family="monospace" font-size="10px">PK,FK mmsi BIGINT</text>
  <text x="600" y="440" font-family="monospace" font-size="10px">timestamp TIMESTAMPTZ</text>
  <text x="600" y="460" font-family="monospace" font-size="10px">geom GEOMETRY(POINT,4326)</text>
  <text x="600" y="480" font-family="monospace" font-size="10px">latitude NUMERIC(9,6)</text>
  <text x="600" y="500" font-family="monospace" font-size="10px">longitude NUMERIC(9,6)</text>
  <text x="600" y="520" font-family="monospace" font-size="10px">sog NUMERIC(5,2)</text>
  <text x="600" y="540" font-family="monospace" font-size="10px">cog NUMERIC(5,2)</text>

  <line x1="250" y1="180" x2="310" y2="180" class="edge-line" marker-end="url(#arrow)" />
  <line x1="530" y1="180" x2="590" y2="180" class="edge-line" marker-end="url(#arrow)" />
  <line x1="420" y1="370" x2="420" y2="330" class="edge-line" stroke-dasharray="4" />
  <line x1="530" y1="460" x2="590" y2="460" class="edge-line" marker-end="url(#arrow)" />
  <line x1="705" y1="370" x2="705" y2="310" class="edge-line" stroke-dasharray="4" />
'''
create_svg("fig3_10_database_er_diagram.svg", 860, 600, "Fig 3.10: PostgreSQL / PostGIS Relational Entity-Relationship Diagram", c_3_10)

# Fig 3.9: Dark-Vessel Candidate Classification Flowchart
c_3_9 = '''
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect x="300" y="60" width="280" height="45" class="box-rect" />
  <text x="440" y="87" class="box-title">SAR Vessel Detection Localized &amp; Georeferenced</text>

  <line x1="440" y1="105" x2="440" y2="140" class="edge-line" marker-end="url(#arrow)" />
  
  <rect x="280" y="140" width="320" height="50" class="box-rect" fill="#f1f5f9" />
  <text x="440" y="162" class="box-title">Spatiotemporal Kinematic Matching</text>
  <text x="440" y="178" class="box-text">Query ±15 min, 4.5km Gate, Compute Uncertainty Budget</text>

  <line x1="440" y1="190" x2="440" y2="230" class="edge-line" marker-end="url(#arrow)" />

  <!-- Decision diamond represented as polygon -->
  <polygon points="440,230 620,280 440,330 260,280" fill="#e0e7ff" stroke="#4338ca" stroke-width="2" />
  <text x="440" y="275" font-family="sans-serif" font-weight="bold" font-size="12px" fill="#312e81" text-anchor="middle">Classification Gate</text>
  <text x="440" y="292" font-family="sans-serif" font-size="10px" fill="#4338ca" text-anchor="middle">Evaluation of z-score and distance</text>

  <!-- 4 Outcome Boxes -->
  <line x1="260" y1="280" x2="120" y2="380" class="edge-line" marker-end="url(#arrow)" />
  <rect x="20" y="380" width="200" height="90" class="status-matched" />
  <text x="120" y="405" class="status-text-matched">AIS-MATCHED</text>
  <text x="120" y="425" class="box-text">Verified Commercial Traffic</text>
  <text x="120" y="440" class="box-text">dist &lt;= 3km and z &lt;= 2.5</text>
  <text x="120" y="455" class="box-text">Record matched MMSI &amp; name</text>

  <line x1="380" y1="320" x2="330" y2="380" class="edge-line" marker-end="url(#arrow)" />
  <rect x="230" y="380" width="200" height="90" class="status-review" />
  <text x="330" y="405" class="status-text-review">AMBIGUOUS / REVIEW</text>
  <text x="330" y="425" class="box-text">Soft-Edge Boundary Candidate</text>
  <text x="330" y="440" class="box-text">2.5 &lt; z &lt;= 4.0 or soft gate</text>
  <text x="330" y="455" class="box-text">Tactical Analyst Review</text>

  <line x1="500" y1="320" x2="550" y2="380" class="edge-line" marker-end="url(#arrow)" />
  <rect x="450" y="380" width="200" height="90" class="status-dark" />
  <text x="550" y="405" class="status-text-dark">AIS-UNMATCHED</text>
  <text x="550" y="425" class="box-text">Potential Dark-Vessel</text>
  <text x="550" y="440" class="box-text">Active AIS coverage in 35km</text>
  <text x="550" y="455" class="box-text">Flag High-Priority Interdiction</text>

  <line x1="620" y1="280" x2="760" y2="380" class="edge-line" marker-end="url(#arrow)" />
  <rect x="660" y="380" width="200" height="90" fill="#f1f5f9" stroke="#64748b" stroke-width="2" rx="6" />
  <text x="760" y="405" font-family="sans-serif" font-weight="bold" font-size="12px" fill="#475569" text-anchor="middle">INSUFFICIENT-COVERAGE</text>
  <text x="760" y="425" class="box-text">Operational Blindspot</text>
  <text x="760" y="440" class="box-text">Zero AIS pings in 35km</text>
  <text x="760" y="455" class="box-text">Non-reporting territory</text>
'''
create_svg("fig3_9_dark_vessel_candidate_classification.svg", 880, 500, "Fig 3.9: Dark-Vessel Candidate Ethical Classification Flowchart", c_3_9)

print("Rendered foundational vector SVGs successfully.")
