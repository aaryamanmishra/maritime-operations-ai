import pytest
from datetime import datetime, timezone
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.infrastructure.database.session import async_session_factory
from app.infrastructure.database.repositories.vessel_repository import VesselRepository
from app.domain.vessel_traffic.models import NormalizedVesselEvent


@pytest.mark.asyncio
async def test_vessel_api_lifecycle():
    # 1. Insert a vessel into the database using repository
    event1 = NormalizedVesselEvent(
        mmsi=999123456,
        timestamp=datetime.now(timezone.utc),
        latitude=50.55,
        longitude=1.25,
        sog=12.5,
        cog=180.0,
        heading=179.0,
        nav_status=0,
        ship_name="TEST CONTAINER",
        imo=9123456,
        callsign="TEST1",
        ship_type=70,
        destination="LE HAVRE",
        length=200.0,
        beam=32.0,
        draught=11.2,
    )

    async with async_session_factory() as session:
        repo = VesselRepository(session)
        await repo.upsert_vessel(event1)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 2. Query bounding box that contains the vessel
        resp = await client.get("/api/v1/vessels", params={
            "min_lat": 50.0,
            "min_lon": 1.0,
            "max_lat": 51.0,
            "max_lon": 2.0,
        })
        assert resp.status_code == 200
        vessels = resp.json()
        matching = [v for v in vessels if v["mmsi"] == 999123456]
        assert len(matching) == 1
        assert matching[0]["name"] == "TEST CONTAINER"
        assert matching[0]["latitude"] == 50.55
        assert matching[0]["longitude"] == 1.25
        assert matching[0]["sog"] == 12.5

        # 3. Query bounding box outside the vessel coordinates
        resp_out = await client.get("/api/v1/vessels", params={
            "min_lat": 40.0,
            "min_lon": 5.0,
            "max_lat": 41.0,
            "max_lon": 6.0,
        })
        assert resp_out.status_code == 200
        assert not any(v["mmsi"] == 999123456 for v in resp_out.json())

        # 4. Query vessel details by MMSI
        resp_details = await client.get("/api/v1/vessels/999123456")
        assert resp_details.status_code == 200
        details = resp_details.json()
        assert details["mmsi"] == 999123456
        assert details["imo"] == 9123456
        assert details["destination"] == "LE HAVRE"
        assert isinstance(details["recent_track"], list)

        # 5. Query non-existent vessel MMSI
        resp_404 = await client.get("/api/v1/vessels/100000001")
        assert resp_404.status_code == 404

        # 6. Pipeline status endpoint
        resp_status = await client.get("/api/v1/vessels/pipeline/status")
        assert resp_status.status_code == 200
        status_data = resp_status.json()
        assert "connection_state" in status_data
        assert "total_vessels_in_db" in status_data
        assert status_data["total_vessels_in_db"] >= 1
