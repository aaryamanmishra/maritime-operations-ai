import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logging import logger
from app.domain.vessel_traffic.models import BoundingBox
from app.infrastructure.redis.client import get_redis_client

ws_router = APIRouter(tags=["WebSocket"])


@ws_router.websocket("/ws/vessels")
async def websocket_vessel_stream(websocket: WebSocket):
    """
    WebSocket endpoint delivering real-time vessel updates to the browser.
    Clients transmit viewport bounding box to filter outgoing updates.
    """
    await websocket.accept()
    logger.info("New WebSocket client connected to /ws/vessels")

    client_bbox: BoundingBox | None = None
    redis = get_redis_client()
    pubsub = redis.pubsub()
    await pubsub.subscribe(settings.AIS_BROADCAST_CHANNEL)

    async def client_reader():
        nonlocal client_bbox
        try:
            while True:
                msg_text = await websocket.receive_text()
                try:
                    payload = json.loads(msg_text)
                    msg_type = payload.get("type")
                    if msg_type == "viewport":
                        # Expected format: [min_lon, min_lat, max_lon, max_lat]
                        bbox_coords = payload.get("bbox")
                        if isinstance(bbox_coords, (list, tuple)) and len(bbox_coords) == 4:
                            min_lon, min_lat, max_lon, max_lat = bbox_coords
                            client_bbox = BoundingBox(
                                min_lat=float(min_lat),
                                min_lon=float(min_lon),
                                max_lat=float(max_lat),
                                max_lon=float(max_lon),
                            )
                            # Acknowledge viewport update
                            await websocket.send_json({
                                "type": "viewport_ack",
                                "bbox": [min_lon, min_lat, max_lon, max_lat]
                            })
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception as e:
                    logger.debug(f"Invalid WebSocket client frame: {e}")
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WebSocket client reader error: {e}")

    async def broadcast_forwarder():
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    raw_data = message["data"]
                    if not client_bbox:
                        # Client has not yet defined a viewport; skip
                        continue

                    try:
                        event_dict = json.loads(raw_data)
                        lat = event_dict.get("latitude")
                        lon = event_dict.get("longitude")

                        if lat is not None and lon is not None:
                            if client_bbox.contains(lat=float(lat), lon=float(lon)):
                                await websocket.send_text(raw_data)
                    except Exception as e:
                        logger.debug(f"Error filtering broadcast frame: {e}")
        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Broadcast forwarder closed: {e}")

    reader_task = asyncio.create_task(client_reader())
    forwarder_task = asyncio.create_task(broadcast_forwarder())

    try:
        done, pending = await asyncio.wait(
            [reader_task, forwarder_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        await pubsub.unsubscribe(settings.AIS_BROADCAST_CHANNEL)
        await pubsub.aclose()
        logger.info("WebSocket client disconnected from /ws/vessels")
