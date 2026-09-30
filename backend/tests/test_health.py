import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch
from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["project"] == "Maritime Operations AI"
        assert data["status"] == "operational"


@pytest.mark.asyncio
async def test_health_live_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "X-Correlation-ID" in response.headers


@pytest.mark.asyncio
async def test_health_ready_healthy():
    with patch("app.api.v1.health.check_database_connection", return_value=True), \
         patch("app.api.v1.health.check_redis_connection", return_value=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["database"] == "ok"
            assert data["redis"] == "ok"


@pytest.mark.asyncio
async def test_health_ready_degraded():
    with patch("app.api.v1.health.check_database_connection", return_value=False), \
         patch("app.api.v1.health.check_redis_connection", return_value=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health/ready")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"
            assert data["database"] == "unavailable"
            assert data["redis"] == "ok"
