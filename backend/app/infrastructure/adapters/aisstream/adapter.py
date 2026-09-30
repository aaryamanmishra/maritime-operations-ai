import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from app.core.logging import logger
from app.domain.vessel_traffic.models import NormalizedVesselEvent


class AISStreamAdapter:
    """
    Translates raw JSON frames received from AISStream WebSocket into
    canonical NormalizedVesselEvent domain models.
    """

    @staticmethod
    def parse_time_utc(time_str: Optional[str]) -> datetime:
        if not time_str:
            return datetime.now(timezone.utc)
        try:
            # Format: '2026-09-30 07:50:20.546648276 +0000 UTC'
            # Strip nanoseconds/UTC suffix for standard parsing
            parts = time_str.split(".")
            if len(parts) == 2:
                sec_part = parts[0]
                return datetime.strptime(sec_part, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)

    @classmethod
    def normalize_message(cls, raw_data: Dict[str, Any]) -> Optional[NormalizedVesselEvent]:
        """
        Normalizes a raw AISStream JSON frame into a NormalizedVesselEvent.
        Returns None if message is a system confirmation or invalid.
        """
        msg_type = raw_data.get("MessageType")
        if not msg_type or msg_type == "SubscriptionConfirmation":
            return None

        metadata = raw_data.get("MetaData", {})
        message_body = raw_data.get("Message", {})

        mmsi = metadata.get("MMSI")
        if not mmsi:
            return None

        time_utc = cls.parse_time_utc(metadata.get("time_utc"))
        ship_name = metadata.get("ShipName")
        meta_lat = metadata.get("latitude")
        meta_lon = metadata.get("longitude")

        # 1. Position Report (Class A - Types 1, 2, 3)
        if msg_type == "PositionReport" and "PositionReport" in message_body:
            pos = message_body["PositionReport"]
            lat = pos.get("Latitude", meta_lat)
            lon = pos.get("Longitude", meta_lon)
            if lat is None or lon is None:
                return None

            return NormalizedVesselEvent(
                mmsi=int(mmsi),
                timestamp=time_utc,
                latitude=float(lat),
                longitude=float(lon),
                sog=float(pos["Sog"]) if pos.get("Sog") is not None else None,
                cog=float(pos["Cog"]) if pos.get("Cog") is not None else None,
                heading=float(pos["TrueHeading"]) if pos.get("TrueHeading") is not None else None,
                nav_status=int(pos["NavigationalStatus"]) if pos.get("NavigationalStatus") is not None else None,
                ship_name=ship_name,
            )

        # 2. Standard Class B Position Report (Type 18)
        elif msg_type == "StandardClassBPositionReport" and "StandardClassBPositionReport" in message_body:
            pos = message_body["StandardClassBPositionReport"]
            lat = pos.get("Latitude", meta_lat)
            lon = pos.get("Longitude", meta_lon)
            if lat is None or lon is None:
                return None

            return NormalizedVesselEvent(
                mmsi=int(mmsi),
                timestamp=time_utc,
                latitude=float(lat),
                longitude=float(lon),
                sog=float(pos["Sog"]) if pos.get("Sog") is not None else None,
                cog=float(pos["Cog"]) if pos.get("Cog") is not None else None,
                heading=float(pos["TrueHeading"]) if pos.get("TrueHeading") is not None else None,
                ship_name=ship_name,
            )

        # 3. Ship Static Data (Type 5 - Voyage & Static metadata)
        elif msg_type == "ShipStaticData" and "ShipStaticData" in message_body:
            stat = message_body["ShipStaticData"]
            dim = stat.get("Dimension", {})
            length = (dim.get("A", 0) + dim.get("B", 0)) if dim else None
            beam = (dim.get("C", 0) + dim.get("D", 0)) if dim else None

            return NormalizedVesselEvent(
                mmsi=int(mmsi),
                timestamp=time_utc,
                latitude=meta_lat,
                longitude=meta_lon,
                ship_name=stat.get("Name") or ship_name,
                imo=stat.get("ImoNumber"),
                callsign=stat.get("CallSign"),
                ship_type=stat.get("Type"),
                destination=stat.get("Destination"),
                length=float(length) if length and length > 0 else None,
                beam=float(beam) if beam and beam > 0 else None,
                draught=float(stat["MaximumStaticDraught"]) if stat.get("MaximumStaticDraught") else None,
            )

        return None
