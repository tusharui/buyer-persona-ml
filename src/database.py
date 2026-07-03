from urllib.parse import urlencode, parse_qs, urlparse, urlunparse

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy import text
from src.config import DATABASE_URL


def _build_async_uri(sync_uri: str) -> str:
    parsed = urlparse(sync_uri.replace("postgresql://", "postgresql+asyncpg://"))
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs.pop("sslmode", None)
    qs.pop("channel_binding", None)
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


engine = create_async_engine(
    _build_async_uri(DATABASE_URL),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


async def check_health() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[db] health check failed: {e}")
        return False


async def close():
    await engine.dispose()
