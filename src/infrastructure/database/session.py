"""
Session PostgreSQL Asynchrone avec SQLAlchemy 2.0 et asyncpg.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ...config.settings import settings
from .models import Base

# Moteur asynchrone pour PostgreSQL
engine = create_async_engine(
    settings.database_url,
    echo=settings.is_development,
    future=True,
    pool_pre_ping=True,
)

# Fabrique de sessions asynchrones
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Générateur de session pour l'injection de dépendances (ex: FastAPI ou Use Cases)."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Crée toutes les tables dans PostgreSQL si elles n'existent pas encore."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Ferme proprement le pool de connexions du moteur."""
    await engine.dispose()
