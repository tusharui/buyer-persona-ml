import json
import hashlib
from typing import Optional, Any

import pandas as pd
import redis.asyncio as aioredis

from src.config import REDIS_URL


class DashboardCache:
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self._client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await self._client.ping()
        except Exception as e:
            print(f"[dashboard-cache] Redis unavailable: {e}")
            self._client = None

    def _key(self, prefix: str, data: Any) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"dashboard:{prefix}:{h}"

    async def get_dataframe(self, prefix: str, params: Any) -> Optional[pd.DataFrame]:
        if not self._client:
            return None
        key = self._key(prefix, params)
        try:
            data = await self._client.get(key)
            if data:
                return pd.read_json(data)
        except Exception:
            pass
        return None

    async def set_dataframe(self, prefix: str, params: Any, df: pd.DataFrame, ttl: int = 3600):
        if not self._client:
            return
        key = self._key(prefix, params)
        try:
            await self._client.setex(key, ttl, df.to_json())
        except Exception:
            pass

    async def get_figure(self, prefix: str, params: Any) -> Optional[bytes]:
        if not self._client:
            return None
        key = self._key(prefix, params)
        try:
            data = await self._client.get(key)
            if data:
                return data.encode("latin-1") if isinstance(data, str) else data
        except Exception:
            pass
        return None

    async def set_figure(self, prefix: str, params: Any, fig_bytes: bytes, ttl: int = 3600):
        if not self._client:
            return
        key = self._key(prefix, params)
        try:
            await self._client.setex(key, ttl, fig_bytes.decode("latin-1"))
        except Exception:
            pass

    async def close(self):
        if self._client:
            await self._client.close()


dash_cache = DashboardCache()
