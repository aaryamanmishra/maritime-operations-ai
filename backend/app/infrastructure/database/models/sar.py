from datetime import UTC, datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .vessel import Base


class SARScene(Base):
    __tablename__ = "sar_scene"

    scene_id = Column(String(128), primary_key=True)
    acquisition_time = Column(DateTime(timezone=True), nullable=False, index=True)
    platform = Column(String(64), nullable=False)  # e.g., "Sentinel-1A"
    orbit_pass = Column(String(32), nullable=True)  # "DESCENDING" / "ASCENDING"
    orbit_number = Column(Integer, nullable=True)
    footprint = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    processing_level = Column(String(32), nullable=False, default="LEVEL1_GRD")
    polarization = Column(String(32), nullable=False, default="VV+VH")
    source = Column(String(64), nullable=False, default="Copernicus CDSE")
    scene_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    # Relationships
    detections = relationship("SARDetection", back_populates="scene", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_sar_scene_footprint", "footprint", postgresql_using="gist"),
        Index("idx_sar_scene_acquisition_time", "acquisition_time"),
    )


class SARDetection(Base):
    __tablename__ = "sar_detection"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    detection_id = Column(String(64), nullable=False, unique=True, index=True)
    scene_id = Column(String(128), ForeignKey("sar_scene.scene_id", ondelete="CASCADE"), nullable=False, index=True)
    geom = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    latitude = Column(Numeric(9, 6), nullable=False)
    longitude = Column(Numeric(9, 6), nullable=False)
    pixel_bbox = Column(JSONB, nullable=True)  # {"ymin": 100, "xmin": 50, "ymax": 140, "xmax": 90}
    confidence = Column(Numeric(4, 3), nullable=False)
    model_version = Column(String(64), nullable=False)
    length_m = Column(Numeric(6, 2), nullable=True)
    heading_deg = Column(Numeric(5, 2), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    # Relationships
    scene = relationship("SARScene", back_populates="detections")
    correlation = relationship("AISSARCorrelation", back_populates="detection", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_sar_detection_geom", "geom", postgresql_using="gist"),
        Index("idx_sar_detection_scene", "scene_id"),
    )


class AISSARCorrelation(Base):
    __tablename__ = "ais_sar_correlation"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    sar_detection_id = Column(BigInteger, ForeignKey("sar_detection.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    candidate_mmsi = Column(BigInteger, nullable=True, index=True)
    time_difference_seconds = Column(Numeric(8, 2), nullable=True)
    distance_km = Column(Numeric(8, 3), nullable=True)
    match_status = Column(String(32), nullable=False)  # "AIS-MATCHED" | "AIS-UNMATCHED"
    matching_method = Column(String(64), nullable=False, default="spatiotemporal_nearest")
    provenance = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    # Relationships
    detection = relationship("SARDetection", back_populates="correlation")

    __table_args__ = (
        Index("idx_ais_sar_correlation_mmsi", "candidate_mmsi"),
        Index("idx_ais_sar_correlation_status", "match_status"),
    )
