"""List research projects through the repository port."""

from __future__ import annotations

from ...domain.ports.research_project_repository import ResearchProjectRepository
from ..dto.research_dtos import ListProjectsInput, ResearchProjectResponseDTO


class ListProjectsUseCase:
    def __init__(self, project_repo: ResearchProjectRepository) -> None:
        self.project_repo = project_repo

    async def execute(
        self, dto: ListProjectsInput | None = None
    ) -> list[ResearchProjectResponseDTO]:
        request = dto or ListProjectsInput()
        projects = await self.project_repo.list_projects(limit=request.limit, offset=request.offset)
        return [
            ResearchProjectResponseDTO(
                id=project.id,
                title=project.title,
                topic=project.topic,
                status=project.status,
                created_by=project.created_by,
                protocol_id=project.protocol_id,
            )
            for project in projects
        ]
