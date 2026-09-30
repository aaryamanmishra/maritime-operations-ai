import pytest
from unittest.mock import AsyncMock, patch
from app.infrastructure.database.session import check_database_connection
from app.infrastructure.redis.client import check_redis_connection


@pytest.mark.asyncio
async def test_live_database_connection():
    # Tests real connectivity against running Postgres container
    result = await check_database_connection()
    assert result is True


@pytest.mark.asyncio
async def test_live_redis_connection():
    # Tests real connectivity against running Redis container
    result = await check_redis_connection()
    assert result is True


@pytest.mark.asyncio
async def test_check_database_connection_failure():
    # When engine raises an error, function returns False gracefully
    with patch("app.infrastructure.database.session.engine") as mock_engine:
        mock_engine.connect.side_effect = Exception("DB connection refused")
        result = await check_database_connection()
        assert result is False


@pytest.mark.asyncio
async def test_check_redis_connection_failure():
    # When redis ping raises an exception, function returns False gracefully
    with patch("app.infrastructure.redis.client.get_redis_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Redis unreachable")
        mock_get.return_value = mock_client

        result = await check_redis_connection()
        assert result is False
