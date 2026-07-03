import json
from typing import Optional
import redis.asyncio as aioredis

from src.config import REDIS_URL, PREDICTION_CACHE_TTL, FEATURE_CACHE_TTL


class CacheClient:
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self._client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await self._client.ping()
        except Exception as e:
            print(f"[cache] Redis unavailable: {e}")
            self._client = None

    async def get(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = PREDICTION_CACHE_TTL):
        if not self._client:
            return
        try:
            await self._client.setex(key, ttl, value)
        except Exception:
            pass

    async def delete(self, key: str):
        if not self._client:
            return
        try:
            await self._client.delete(key)
        except Exception:
            pass

    async def exists(self, key: str) -> bool:
        if not self._client:
            return False
        try:
            return bool(await self._client.exists(key))
        except Exception:
            return False

    async def incr(self, key: str) -> int:
        if not self._client:
            return 0
        try:
            return await self._client.incr(key)
        except Exception:
            return 0

    async def expire(self, key: str, ttl: int):
        if not self._client:
            return
        try:
            await self._client.expire(key, ttl)
        except Exception:
            pass

    async def close(self):
        if self._client:
            await self._client.close()

    @property
    def prediction_cache_ttl(self) -> int:
        return PREDICTION_CACHE_TTL

    @property
    def feature_cache_ttl(self) -> int:
        return FEATURE_CACHE_TTL


cache = CacheClient()
