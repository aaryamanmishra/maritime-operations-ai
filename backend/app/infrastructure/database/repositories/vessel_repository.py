from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.vessel_traffic.models import (
    BoundingBox,
    NormalizedVesselEvent,
    VesselCurrentState,
    VesselDetails,
    VesselPositionPoint,
    should_record_history,
)


class VesselRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_vessel(self, event: NormalizedVesselEvent) -> bool:
        """
        Upserts vessel identity, updates current state if coordinates exist,
        and applies the deadband filter to store historical trajectory.
        """
        # 1. Upsert vessel_identity
        identity_sql = text("""
            INSERT INTO vessel_identity (
                mmsi, imo, name, callsign, ship_type, length, beam, draught, destination, updated_at
            ) VALUES (
                :mmsi, :imo, :name, :callsign, :ship_type, :length, :beam, :draught, :destination, :updated_at
            )
            ON CONFLICT (mmsi) DO UPDATE SET
                imo = COALESCE(EXCLUDED.imo, vessel_identity.imo),
                name = COALESCE(EXCLUDED.name, vessel_identity.name),
                callsign = COALESCE(EXCLUDED.callsign, vessel_identity.callsign),
                ship_type = COALESCE(EXCLUDED.ship_type, vessel_identity.ship_type),
                length = COALESCE(EXCLUDED.length, vessel_identity.length),
                beam = COALESCE(EXCLUDED.beam, vessel_identity.beam),
                draught = COALESCE(EXCLUDED.draught, vessel_identity.draught),
                destination = COALESCE(EXCLUDED.destination, vessel_identity.destination),
                updated_at = EXCLUDED.updated_at;
        """)

        await self.session.execute(identity_sql, {
            "mmsi": event.mmsi,
            "imo": event.imo,
            "name": event.ship_name,
            "callsign": event.callsign,
            "ship_type": event.ship_type,
            "length": event.length,
            "beam": event.beam,
            "draught": event.draught,
            "destination": event.destination,
            "updated_at": event.timestamp,
        })

        # If no coordinates in event, finish
        if event.latitude is None or event.longitude is None:
            await self.session.commit()
            return True

        # 2. Check last history point for thinning / deadband filter
        last_hist_sql = text("""
            SELECT latitude, longitude, heading, timestamp
            FROM vessel_position_history
            WHERE mmsi = :mmsi
            ORDER BY timestamp DESC
            LIMIT 1;
        """)
        last_hist_res = (await self.session.execute(last_hist_sql, {"mmsi": event.mmsi})).first()
        last_lat = float(last_hist_res[0]) if last_hist_res else None
        last_lon = float(last_hist_res[1]) if last_hist_res else None
        last_heading = float(last_hist_res[2]) if last_hist_res and last_hist_res[2] is not None else None
        last_timestamp = last_hist_res[3] if last_hist_res else None

        # 3. Upsert current_vessel_state
        state_sql = text("""
            INSERT INTO current_vessel_state (
                mmsi, timestamp, geom, latitude, longitude, sog, cog, heading, nav_status, updated_at
            ) VALUES (
                :mmsi, :timestamp, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                :lat, :lon, :sog, :cog, :heading, :nav_status, NOW()
            )
            ON CONFLICT (mmsi) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                geom = EXCLUDED.geom,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                sog = COALESCE(EXCLUDED.sog, current_vessel_state.sog),
                cog = COALESCE(EXCLUDED.cog, current_vessel_state.cog),
                heading = COALESCE(EXCLUDED.heading, current_vessel_state.heading),
                nav_status = COALESCE(EXCLUDED.nav_status, current_vessel_state.nav_status),
                updated_at = NOW();
        """)

        await self.session.execute(state_sql, {
            "mmsi": event.mmsi,
            "timestamp": event.timestamp,
            "lat": event.latitude,
            "lon": event.longitude,
            "sog": event.sog,
            "cog": event.cog,
            "heading": event.heading,
            "nav_status": event.nav_status,
        })

        # 4. Insert into vessel_position_history if thinning criteria are met
        record_history = should_record_history(
            last_lat=last_lat,
            last_lon=last_lon,
            last_heading=last_heading,
            last_timestamp=last_timestamp,
            new_event=event,
            distance_threshold_m=settings.AIS_HISTORY_DISTANCE_THRESHOLD_M,
            heading_threshold_deg=settings.AIS_HISTORY_HEADING_THRESHOLD_DEG,
            time_threshold_s=settings.AIS_HISTORY_TIME_THRESHOLD_S,
        )

        if record_history:
            history_sql = text("""
                INSERT INTO vessel_position_history (
                    mmsi, timestamp, geom, latitude, longitude, sog, cog, heading, nav_status
                ) VALUES (
                    :mmsi, :timestamp, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                    :lat, :lon, :sog, :cog, :heading, :nav_status
                );
            """)
            await self.session.execute(history_sql, {
                "mmsi": event.mmsi,
                "timestamp": event.timestamp,
                "lat": event.latitude,
                "lon": event.longitude,
                "sog": event.sog,
                "cog": event.cog,
                "heading": event.heading,
                "nav_status": event.nav_status,
            })

        await self.session.commit()
        return True

    async def get_vessels_in_bbox(self, bbox: BoundingBox, limit: int = 500) -> list[VesselCurrentState]:
        """
        Spatial query using PostGIS GiST index to fetch current vessel states within a bounding box.
        """
        query = text("""
            SELECT c.mmsi, i.name, i.imo, i.callsign, i.ship_type, i.destination,
                   i.length, i.beam, i.draught,
                   c.latitude, c.longitude, c.sog, c.cog, c.heading, c.nav_status,
                   c.timestamp, c.updated_at
            FROM current_vessel_state c
            JOIN vessel_identity i ON c.mmsi = i.mmsi
            WHERE c.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
            ORDER BY c.updated_at DESC
            LIMIT :limit;
        """)

        result = await self.session.execute(query, {
            "min_lon": bbox.min_lon,
            "min_lat": bbox.min_lat,
            "max_lon": bbox.max_lon,
            "max_lat": bbox.max_lat,
            "limit": limit,
        })

        rows = result.fetchall()
        vessels = []
        for r in rows:
            vessels.append(VesselCurrentState(
                mmsi=r[0],
                name=r[1],
                imo=r[2],
                callsign=r[3],
                ship_type=r[4],
                destination=r[5],
                length=float(r[6]) if r[6] is not None else None,
                beam=float(r[7]) if r[7] is not None else None,
                draught=float(r[8]) if r[8] is not None else None,
                latitude=float(r[9]),
                longitude=float(r[10]),
                sog=float(r[11]) if r[11] is not None else None,
                cog=float(r[12]) if r[12] is not None else None,
                heading=float(r[13]) if r[13] is not None else None,
                nav_status=r[14],
                timestamp=r[15],
                updated_at=r[16],
            ))
        return vessels

    async def get_vessel_details(self, mmsi: int, history_limit: int = 50) -> VesselDetails | None:
        """
        Fetches full details for a vessel plus its recent persisted track points.
        """
        current_sql = text("""
            SELECT c.mmsi, i.name, i.imo, i.callsign, i.ship_type, i.destination,
                   i.length, i.beam, i.draught,
                   c.latitude, c.longitude, c.sog, c.cog, c.heading, c.nav_status,
                   c.timestamp, c.updated_at
            FROM current_vessel_state c
            JOIN vessel_identity i ON c.mmsi = i.mmsi
            WHERE c.mmsi = :mmsi;
        """)

        res = (await self.session.execute(current_sql, {"mmsi": mmsi})).first()
        if not res:
            return None

        # Fetch recent track (ordered chronologically)
        hist_sql = text("""
            SELECT latitude, longitude, timestamp, sog, cog, heading
            FROM vessel_position_history
            WHERE mmsi = :mmsi
            ORDER BY timestamp ASC
            LIMIT :history_limit;
        """)

        hist_res = await self.session.execute(hist_sql, {"mmsi": mmsi, "history_limit": history_limit})
        track_points = []
        for h in hist_res.fetchall():
            track_points.append(VesselPositionPoint(
                latitude=float(h[0]),
                longitude=float(h[1]),
                timestamp=h[2],
                sog=float(h[3]) if h[3] is not None else None,
                cog=float(h[4]) if h[4] is not None else None,
                heading=float(h[5]) if h[5] is not None else None,
            ))

        return VesselDetails(
            mmsi=res[0],
            name=res[1],
            imo=res[2],
            callsign=res[3],
            ship_type=res[4],
            destination=res[5],
            length=float(res[6]) if res[6] is not None else None,
            beam=float(res[7]) if res[7] is not None else None,
            draught=float(res[8]) if res[8] is not None else None,
            latitude=float(res[9]),
            longitude=float(res[10]),
            sog=float(res[11]) if res[11] is not None else None,
            cog=float(res[12]) if res[12] is not None else None,
            heading=float(res[13]) if res[13] is not None else None,
            nav_status=res[14],
            timestamp=res[15],
            updated_at=res[16],
            recent_track=track_points,
        )

    async def count_total_vessels(self) -> int:
        query = text("SELECT COUNT(*) FROM current_vessel_state;")
        result = await self.session.execute(query)
        return int(result.scalar() or 0)
