"""create sar tables

Revision ID: 0003_create_sar_tables
Revises: 0002_create_vessel_tables
Create Date: 2026-09-30 14:45:00.000000

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_create_sar_tables"
down_revision: Union[str, None] = "0002_create_vessel_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. sar_scene
    op.execute("""
    CREATE TABLE IF NOT EXISTS sar_scene (
        scene_id VARCHAR(128) PRIMARY KEY,
        acquisition_time TIMESTAMPTZ NOT NULL,
        platform VARCHAR(64) NOT NULL,
        orbit_pass VARCHAR(32),
        orbit_number INTEGER,
        footprint GEOMETRY(Polygon, 4326) NOT NULL,
        processing_level VARCHAR(32) NOT NULL DEFAULT 'LEVEL1_GRD',
        polarization VARCHAR(32) NOT NULL DEFAULT 'VV+VH',
        source VARCHAR(64) NOT NULL DEFAULT 'Copernicus CDSE',
        scene_metadata JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_sar_scene_footprint ON sar_scene USING GIST (footprint);
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_sar_scene_acquisition_time ON sar_scene (acquisition_time DESC);
    """)

    # 2. sar_detection
    op.execute("""
    CREATE TABLE IF NOT EXISTS sar_detection (
        id BIGSERIAL PRIMARY KEY,
        detection_id VARCHAR(64) UNIQUE NOT NULL,
        scene_id VARCHAR(128) NOT NULL REFERENCES sar_scene(scene_id) ON DELETE CASCADE,
        geom GEOMETRY(Point, 4326) NOT NULL,
        latitude NUMERIC(9, 6) NOT NULL,
        longitude NUMERIC(9, 6) NOT NULL,
        pixel_bbox JSONB,
        confidence NUMERIC(4, 3) NOT NULL,
        model_version VARCHAR(64) NOT NULL,
        length_m NUMERIC(6, 2),
        heading_deg NUMERIC(5, 2),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_sar_detection_geom ON sar_detection USING GIST (geom);
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_sar_detection_scene ON sar_detection (scene_id);
    """)

    # 3. ais_sar_correlation
    op.execute("""
    CREATE TABLE IF NOT EXISTS ais_sar_correlation (
        id BIGSERIAL PRIMARY KEY,
        sar_detection_id BIGINT UNIQUE NOT NULL REFERENCES sar_detection(id) ON DELETE CASCADE,
        candidate_mmsi BIGINT,
        time_difference_seconds NUMERIC(8, 2),
        distance_km NUMERIC(8, 3),
        match_status VARCHAR(32) NOT NULL,
        matching_method VARCHAR(64) NOT NULL DEFAULT 'spatiotemporal_nearest',
        provenance JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_ais_sar_correlation_mmsi ON ais_sar_correlation (candidate_mmsi);
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_ais_sar_correlation_status ON ais_sar_correlation (match_status);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ais_sar_correlation CASCADE;")
    op.execute("DROP TABLE IF EXISTS sar_detection CASCADE;")
    op.execute("DROP TABLE IF EXISTS sar_scene CASCADE;")
