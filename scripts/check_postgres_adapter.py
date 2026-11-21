"""Run a rolled-back read/write smoke test against the configured PostgreSQL DB."""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.entities.research_project import ResearchProject
from src.infrastructure.persistence.database import create_session_factory
from src.infrastructure.persistence.repositories import ProjectRepo


async def main() -> None:
    sessions = create_session_factory()
    try:
        async with sessions() as session:
            transaction = await session.begin()
            project = ResearchProject(
                id=str(uuid.uuid4()), title="PostgreSQL adapter smoke test",
                topic="Temporary transaction used to verify PostgreSQL persistence", created_by="smoke-test",
            )
            repo = ProjectRepo(session)
            saved = await repo.save(project)
            loaded = await repo.find_by_id(project.id)
            if saved.id != project.id or loaded is None or loaded.title != project.title:
                raise RuntimeError("PostgreSQL project write/read round-trip failed")
            await transaction.rollback()
        print("PostgreSQL OK: project write/read round-trip passed; test transaction rolled back.")
    finally:
        await sessions.kw["bind"].dispose()


if __name__ == "__main__":
    asyncio.run(main())
