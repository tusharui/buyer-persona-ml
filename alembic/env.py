import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from urllib.parse import urlencode, parse_qs, urlparse, urlunparse

from src.config import DATABASE_URL
from src.models import Base


def _build_async_uri(sync_uri: str) -> str:
    parsed = urlparse(sync_uri.replace("postgresql://", "postgresql+asyncpg://"))
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs.pop("sslmode", None)
    qs.pop("channel_binding", None)
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", _build_async_uri(DATABASE_URL))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
