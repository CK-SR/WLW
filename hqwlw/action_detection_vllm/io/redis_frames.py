from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Tuple

from redis import asyncio as aioredis

from config.settings import settings


class RedisFrameClient:
    def __init__(self) -> None:
        self.redis = aioredis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            password=None,
            db=0,
            decode_responses=False,
        )

    async def get_frame(
        self, stream_key: str, timeout_ms: int = 1000
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[bytes]]:
        streams = await self.redis.xread({stream_key: "$"}, count=1, block=timeout_ms)
        if not streams:
            return None, None, None
        for _, messages in streams:
            for msg_id, fields in messages:
                meta = json.loads(fields[b"meta"].decode("utf-8"))
                jpeg = fields[b"jpeg"]
                decoded_id = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                return decoded_id, meta, jpeg
        return None, None, None

    async def close(self) -> None:
        await self.redis.aclose()


def stream_key(camera: str) -> str:
    if camera.startswith(settings.redis.stream_prefix):
        return camera
    return f"{settings.redis.stream_prefix}{camera}"
