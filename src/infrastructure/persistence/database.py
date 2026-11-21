"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ...config.settings import settings
from .models import Base


def create_session_factory(database_url: str | None = None) -> async_sessionmaker[AsyncSession]:
    """Create a session factory; passing a URL keeps tests isolated from global settings."""
    engine = create_async_engine(database_url or settings.resolved_database_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session and commit only after a successful unit of work."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


__all__ = ["Base", "create_session_factory", "session_scope"]
