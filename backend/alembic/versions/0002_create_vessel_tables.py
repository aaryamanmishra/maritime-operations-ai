"""create vessel tables

Revision ID: 0002_create_vessel_tables
Revises: 0001_initial_postgis_foundation
Create Date: 2026-09-30 13:20:00.000000

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_create_vessel_tables"
down_revision: Union[str, None] = "0001_initial_postgis_foundation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. vessel_identity: Static & voyage metadata per vessel
    op.execute("""
    CREATE TABLE IF NOT EXISTS vessel_identity (
        mmsi BIGINT PRIMARY KEY,
        imo INTEGER,
        name VARCHAR(128),
        callsign VARCHAR(32),
        ship_type INTEGER,
        length NUMERIC(8, 2),
        beam NUMERIC(8, 2),
        draught NUMERIC(6, 2),
        destination VARCHAR(128),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_vessel_identity_ship_type ON vessel_identity (ship_type);
    """)

    # 2. current_vessel_state: Exactly one row per vessel holding latest live kinematics
    op.execute("""
    CREATE TABLE IF NOT EXISTS current_vessel_state (
        mmsi BIGINT PRIMARY KEY REFERENCES vessel_identity(mmsi) ON DELETE CASCADE,
        timestamp TIMESTAMPTZ NOT NULL,
        geom GEOMETRY(Point, 4326) NOT NULL,
        latitude NUMERIC(9, 6) NOT NULL,
        longitude NUMERIC(9, 6) NOT NULL,
        sog NUMERIC(5, 2),
        cog NUMERIC(5, 2),
        heading NUMERIC(5, 2),
        nav_status INTEGER,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_current_vessel_state_geom ON current_vessel_state USING GIST (geom);
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_current_vessel_state_updated ON current_vessel_state (updated_at DESC);
    """)

    # 3. vessel_position_history: Thinned historical trajectory points
    op.execute("""
    CREATE TABLE IF NOT EXISTS vessel_position_history (
        id BIGSERIAL PRIMARY KEY,
        mmsi BIGINT NOT NULL REFERENCES vessel_identity(mmsi) ON DELETE CASCADE,
        timestamp TIMESTAMPTZ NOT NULL,
        geom GEOMETRY(Point, 4326) NOT NULL,
        latitude NUMERIC(9, 6) NOT NULL,
        longitude NUMERIC(9, 6) NOT NULL,
        sog NUMERIC(5, 2),
        cog NUMERIC(5, 2),
        heading NUMERIC(5, 2),
        nav_status INTEGER
    );
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_vessel_history_geom ON vessel_position_history USING GIST (geom);
    """)
    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_vessel_history_mmsi_time ON vessel_position_history (mmsi, timestamp DESC);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS vessel_position_history CASCADE;")
    op.execute("DROP TABLE IF EXISTS current_vessel_state CASCADE;")
    op.execute("DROP TABLE IF EXISTS vessel_identity CASCADE;")
