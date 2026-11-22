"""PostgreSQL-backed LangGraph checkpointer lifecycle."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from ..config.settings import settings


def configure_postgres_event_loop() -> None:
    """Psycopg async requires SelectorEventLoop on Windows, not the Proactor default."""
    if sys.platform == "win32":
        proactor_policy = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
        if proactor_policy and isinstance(asyncio.get_event_loop_policy(), proactor_policy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


configure_postgres_event_loop()


class PostgreSQLCheckpointer:
    """Own the psycopg pool and LangGraph's official async PostgreSQL saver."""

    def __init__(self, database_url: str | None = None) -> None:
        configure_postgres_event_loop()
        self.database_url = database_url or settings.resolved_database_url
        self.pool: Any = None
        self.saver: Any = None

    async def open(self) -> Any:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "Installez les dépendances de Phase 2 (langgraph-checkpoint-postgres et psycopg[binary,pool])."
            ) from exc
        conninfo = self.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        self.pool = AsyncConnectionPool(
            conninfo=conninfo,
            kwargs={"autocommit": True, "row_factory": dict_row, "connect_timeout": 5},
            timeout=5.0,
            open=False,
        )
        await self.pool.open()
        self.saver = AsyncPostgresSaver(self.pool)
        try:
            await self.saver.setup()
        except Exception:
            await self.close()
            raise
        return self.saver

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            self.saver = None
