import pytest
from app.infrastructure.database.session import engine
from app.infrastructure.redis.client import close_redis_connection


@pytest.fixture(autouse=True)
async def reset_connections():
    yield
    try:
        await close_redis_connection()
    except Exception:
        pass
    try:
        await engine.dispose()
    except Exception:
        pass

