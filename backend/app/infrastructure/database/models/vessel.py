from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger, Column, DateTime, Integer, Numeric, String, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship
from geoalchemy2 import Geometry

Base = declarative_base()


class VesselIdentity(Base):
    __tablename__ = "vessel_identity"

    mmsi = Column(BigInteger, primary_key=True)
    imo = Column(Integer, nullable=True)
    name = Column(String(128), nullable=True)
    callsign = Column(String(32), nullable=True)
    ship_type = Column(Integer, nullable=True)
    length = Column(Numeric(8, 2), nullable=True)
    beam = Column(Numeric(8, 2), nullable=True)
    draught = Column(Numeric(6, 2), nullable=True)
    destination = Column(String(128), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    current_state = relationship("CurrentVesselState", back_populates="identity", uselist=False, cascade="all, delete-orphan")
    history_positions = relationship("VesselPositionHistory", back_populates="identity", cascade="all, delete-orphan")


class CurrentVesselState(Base):
    __tablename__ = "current_vessel_state"

    mmsi = Column(BigInteger, ForeignKey("vessel_identity.mmsi", ondelete="CASCADE"), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    geom = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    latitude = Column(Numeric(9, 6), nullable=False)
    longitude = Column(Numeric(9, 6), nullable=False)
    sog = Column(Numeric(5, 2), nullable=True)
    cog = Column(Numeric(5, 2), nullable=True)
    heading = Column(Numeric(5, 2), nullable=True)
    nav_status = Column(Integer, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    identity = relationship("VesselIdentity", back_populates="current_state")


class VesselPositionHistory(Base):
    __tablename__ = "vessel_position_history"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    mmsi = Column(BigInteger, ForeignKey("vessel_identity.mmsi", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    geom = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    latitude = Column(Numeric(9, 6), nullable=False)
    longitude = Column(Numeric(9, 6), nullable=False)
    sog = Column(Numeric(5, 2), nullable=True)
    cog = Column(Numeric(5, 2), nullable=True)
    heading = Column(Numeric(5, 2), nullable=True)
    nav_status = Column(Integer, nullable=True)

    identity = relationship("VesselIdentity", back_populates="history_positions")
