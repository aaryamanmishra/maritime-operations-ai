
import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger

redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return redis_client


async def close_redis_connection() -> None:
    global redis_client
    if redis_client is not None:
        try:
            await redis_client.aclose()
        except RuntimeError:
            pass
        except Exception as exc:
            logger.warning(f"Error closing Redis connection: {exc}")
        finally:
            redis_client = None
            logger.info("Redis connection closed")


async def check_redis_connection() -> bool:
    try:
        client = get_redis_client()
        pong = await client.ping()
        return bool(pong)
    except Exception as exc:
        logger.warning(f"Redis health check failed: {exc}")
        return False
