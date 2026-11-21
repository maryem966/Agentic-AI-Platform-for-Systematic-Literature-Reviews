"""Port for persisting and listing research projects."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..entities.research_project import ResearchProject


class ResearchProjectRepository(ABC):
    @abstractmethod
    async def save(self, project: ResearchProject) -> ResearchProject: ...

    @abstractmethod
    async def find_by_id(self, project_id: str) -> ResearchProject | None: ...

    @abstractmethod
    async def list_projects(self, limit: int = 100, offset: int = 0) -> list[ResearchProject]: ...
