import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import websockets
from websockets.exceptions import ConnectionClosed

from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.domain.vessel_traffic.models import NormalizedVesselEvent
from app.infrastructure.adapters.aisstream.adapter import AISStreamAdapter
from app.infrastructure.database.session import async_session_factory
from app.infrastructure.database.repositories.vessel_repository import VesselRepository
from app.infrastructure.redis.client import get_redis_client, close_redis_connection


class AISPipelineWorker:
    """
    Dedicated background worker responsible for:
    1. Ingestion: Long-lived WebSocket connection to AISStream.io -> Normalization -> Redis Stream.
    2. Persistence: Redis Consumer Group consumer -> PostGIS upsert & thinning -> Redis Pub/Sub broadcast.
    """

    def __init__(self):
        self._running = False
        self._stop_event = asyncio.Event()
        self._connection_state = "disconnected"  # connected, reconnecting, disconnected
        
        # Metrics
        self._messages_received = 0
        self._normalized_count = 0
        self._stream_writes = 0
        self._persisted_count = 0
        self._malformed_count = 0
        self._last_message_at: Optional[datetime] = None
        self._connected_at: Optional[datetime] = None

    async def start(self):
        self._running = True
        self._stop_event.clear()
        setup_logging()
        logger.info(
            f"Starting AISPipelineWorker for bounding box "
            f"[{settings.AIS_BBOX_MIN_LAT}, {settings.AIS_BBOX_MIN_LON}, "
            f"{settings.AIS_BBOX_MAX_LAT}, {settings.AIS_BBOX_MAX_LON}]"
        )

        # Run ingestion loop, persistence consumer loop, and metrics sync loop concurrently
        tasks = [
            asyncio.create_task(self._ingestion_loop()),
            asyncio.create_task(self._persistence_consumer_loop()),
            asyncio.create_task(self._metrics_sync_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("AISPipelineWorker tasks cancelled, shutting down...")
        finally:
            await self.stop()

    async def stop(self):
        self._running = False
        self._stop_event.set()
        self._connection_state = "disconnected"
        await self._sync_metrics_to_redis()
        await close_redis_connection()
        logger.info("AISPipelineWorker stopped gracefully.")

    async def _ingestion_loop(self):
        """
        Long-lived connection to AISStream with exponential backoff + jitter.
        """
        backoff = 2.0
        max_backoff = 60.0
        backoff_factor = 1.5

        if not settings.AISSTREAM_API_KEY:
            logger.warning("AISSTREAM_API_KEY is not set. AIS ingestion is disabled.")
            self._connection_state = "disconnected"
            while self._running:
                await asyncio.sleep(5)
            return

        while self._running:
            try:
                self._connection_state = "connecting"
                logger.info(f"Connecting to AISStream at {settings.AISSTREAM_URL}...")

                # Connect with permessage-deflate compression enabled
                async with websockets.connect(
                    settings.AISSTREAM_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    compression="deflate",
                    close_timeout=10,
                ) as ws:
                    self._connection_state = "connected"
                    self._connected_at = datetime.now(timezone.utc)
                    backoff = 2.0  # Reset backoff upon successful connection
                    logger.info("AISStream WebSocket connection established successfully.")

                    # Send subscription immediately
                    sub_message = {
                        "APIKey": settings.AISSTREAM_API_KEY,
                        "BoundingBoxes": [
                            [
                                [settings.AIS_BBOX_MIN_LAT, settings.AIS_BBOX_MIN_LON],
                                [settings.AIS_BBOX_MAX_LAT, settings.AIS_BBOX_MAX_LON],
                            ]
                        ],
                        "FilterMessageTypes": [
                            "PositionReport",
                            "StandardClassBPositionReport",
                            "ExtendedClassBPositionReport",
                            "ShipStaticData",
                        ],
                    }
                    await ws.send(json.dumps(sub_message))
                    logger.info("AISStream subscription payload transmitted.")

                    redis = get_redis_client()

                    async for raw_msg in ws:
                        if not self._running:
                            break

                        self._messages_received += 1
                        self._last_message_at = datetime.now(timezone.utc)

                        try:
                            data = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            self._malformed_count += 1
                            continue

                        # Subscription confirmation
                        if data.get("MessageType") == "SubscriptionConfirmation":
                            logger.info("AISStream subscription confirmed by server.")
                            continue

                        # Normalize
                        event = AISStreamAdapter.normalize_message(data)
                        if not event:
                            continue

                        self._normalized_count += 1

                        # Append to Redis Stream (stream:ais:raw)
                        payload = {"data": event.model_dump_json()}
                        await redis.xadd(
                            name=settings.AIS_STREAM_KEY,
                            fields=payload,
                            maxlen=50000,
                            approximate=True,
                        )
                        self._stream_writes += 1

            except ConnectionClosed as exc:
                self._connection_state = "reconnecting"
                logger.warning(f"AISStream connection closed: code={exc.code}, reason={exc.reason}")
            except Exception as exc:
                self._connection_state = "reconnecting"
                logger.error(f"Unexpected AISStream connection error: {exc}")

            if not self._running:
                break

            # Calculate sleep with exponential backoff + jitter
            jitter = random.uniform(0.5, 1.5)
            sleep_time = min(backoff * jitter, max_backoff)
            logger.info(f"Reconnecting to AISStream in {sleep_time:.2f} seconds...")
            await asyncio.sleep(sleep_time)
            backoff = min(backoff * backoff_factor, max_backoff)

    async def _persistence_consumer_loop(self):
        """
        Consumes from Redis Stream `stream:ais:raw` using a Redis Consumer Group,
        persists into PostGIS, and broadcasts to Redis Pub/Sub.
        """
        redis = get_redis_client()

        # Ensure consumer group exists
        try:
            await redis.xgroup_create(
                name=settings.AIS_STREAM_KEY,
                groupname=settings.AIS_CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created Redis Consumer Group [{settings.AIS_CONSUMER_GROUP}].")
        except Exception:
            # Group already exists
            pass

        while self._running:
            try:
                # Read new entries from stream
                messages = await redis.xreadgroup(
                    groupname=settings.AIS_CONSUMER_GROUP,
                    consumername=settings.AIS_CONSUMER_NAME,
                    streams={settings.AIS_STREAM_KEY: ">"},
                    count=100,
                    block=1000,
                )

                if not messages:
                    continue

                for stream_name, entries in messages:
                    for entry_id, fields in entries:
                        if not self._running:
                            break

                        raw_json = fields.get("data")
                        if not raw_json:
                            await redis.xack(settings.AIS_STREAM_KEY, settings.AIS_CONSUMER_GROUP, entry_id)
                            continue

                        try:
                            event_dict = json.loads(raw_json)
                            event = NormalizedVesselEvent(**event_dict)

                            # Persist into PostGIS via async session
                            async with async_session_factory() as session:
                                repo = VesselRepository(session)
                                await repo.upsert_vessel(event)
                                self._persisted_count += 1

                            # Broadcast to Redis Pub/Sub channel for realtime WebSocket clients
                            await redis.publish(settings.AIS_BROADCAST_CHANNEL, raw_json)

                        except Exception as e:
                            logger.error(f"Error persisting/broadcasting AIS event: {e}")
                        finally:
                            # Acknowledge processed message in Redis Stream
                            await redis.xack(settings.AIS_STREAM_KEY, settings.AIS_CONSUMER_GROUP, entry_id)

            except Exception as exc:
                if self._running:
                    logger.error(f"Error in persistence consumer loop: {exc}")
                    await asyncio.sleep(1)

    async def _metrics_sync_loop(self):
        """
        Periodically writes pipeline metrics to Redis for the API to inspect.
        """
        while self._running:
            try:
                await self._sync_metrics_to_redis()
            except Exception as exc:
                logger.warning(f"Error syncing AIS metrics to Redis: {exc}")
            await asyncio.sleep(2)

    async def _sync_metrics_to_redis(self):
        redis = get_redis_client()
        metrics = {
            "connection_state": self._connection_state,
            "messages_received": self._messages_received,
            "normalized_count": self._normalized_count,
            "stream_writes": self._stream_writes,
            "persisted_count": self._persisted_count,
            "malformed_count": self._malformed_count,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "bbox": [
                settings.AIS_BBOX_MIN_LAT,
                settings.AIS_BBOX_MIN_LON,
                settings.AIS_BBOX_MAX_LAT,
                settings.AIS_BBOX_MAX_LON,
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await redis.set(
            settings.AIS_STATUS_REDIS_KEY,
            json.dumps(metrics),
            ex=15,  # 15s TTL ensures status marks stale if worker crashes
        )


async def main():
    worker = AISPipelineWorker()
    try:
        await worker.start()
    except (KeyboardInterrupt, SystemExit):
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
