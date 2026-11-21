from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.application.dto.research_dtos import CreateResearchProjectDTO, DefineProtocolDTO, ResearchQuestionDTO
from src.application.use_cases.create_research_project import CreateResearchProjectUseCase
from src.application.use_cases.define_protocol import DefineProtocolUseCase
from src.application.use_cases.list_projects import ListProjectsUseCase
from src.domain.entities.research_project import ResearchProject
from src.domain.entities.research_protocol import ResearchProtocol, ResearchQuestion
from src.infrastructure.persistence.models import Base
from src.infrastructure.persistence.repositories import AuditLogRepo, ProjectRepo, ProtocolRepo


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.mark.asyncio
async def test_project_repository_and_use_cases_persist(session_factory) -> None:
    async with session_factory() as session:
        projects = ProjectRepo(session)
        created = await CreateResearchProjectUseCase(projects).execute(
            CreateResearchProjectDTO(
                title="Fraud detection review", topic="Machine learning for financial fraud", created_by="test"
            )
        )
        await session.commit()
        assert await projects.find_by_id(created.id)

        listed = await ListProjectsUseCase(projects).execute()
        assert [item.id for item in listed] == [created.id]

        protocol_repo = ProtocolRepo(session)
        protocol = await DefineProtocolUseCase(protocol_repo).execute(DefineProtocolDTO(
            research_id=created.id,
            research_questions=[ResearchQuestionDTO(text="Which methods detect financial fraud?")],
            inclusion_criteria=["peer reviewed", "English language"],
            exclusion_criteria=["not empirical", "not fraud focused"],
            defined_by="test",
        ))
        await session.commit()
        assert protocol.research_id == created.id
        assert (await protocol_repo.find_latest_by_research(created.id)).id == protocol.id
        assert (await projects.find_by_id(created.id)).protocol_id == protocol.id


@pytest.mark.asyncio
async def test_audit_log_is_append_only_in_repository_and_orm(session_factory) -> None:
    async with session_factory() as session:
        repo = AuditLogRepo(session)
        entry = await repo.append(
            audit_id="audit-1", actor="reviewer", action="project.created",
            resource_type="project", resource_id="p-1", payload={"title": "SLR"},
        )
        await session.commit()
        assert (await repo.list_for_resource("project", "p-1"))[0].id == entry.id
        entry.action = "changed"
        with pytest.raises(ValueError, match="immutable"):
            await session.flush()


def test_domain_models_are_accepted_by_repo_contracts() -> None:
    assert ResearchProject(title="Fraud review", topic="Fraud detection methods", created_by="test")
    assert ResearchProtocol(
        research_id="r-1", research_questions=[ResearchQuestion(text="How is fraud detected?")]
    )
