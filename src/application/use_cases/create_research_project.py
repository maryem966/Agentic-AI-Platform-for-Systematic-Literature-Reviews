"""
Use Case : CreateResearchProject
Crée un nouveau projet SLR.
"""

from __future__ import annotations

from ...domain.entities.research_project import ResearchProject
from ...domain.ports.research_project_repository import ResearchProjectRepository
from ..dto.research_dtos import CreateResearchProjectDTO, ResearchProjectResponseDTO


class CreateResearchProjectUseCase:
    """
    Use case : Créer un nouveau projet SLR.

    Persiste le projet via le port injecté, sans dépendre d'un adapter concret.
    """

    def __init__(self, project_repo: ResearchProjectRepository) -> None:
        self.project_repo = project_repo

    async def execute(self, dto: CreateResearchProjectDTO) -> ResearchProjectResponseDTO:
        """
        Crée un ResearchProject à partir du DTO.
        Validation Pydantic automatique par le modèle de domaine.
        """
        project = ResearchProject(
            title=dto.title,
            topic=dto.topic,
            created_by=dto.created_by,
        )

        project = await self.project_repo.save(project)

        return ResearchProjectResponseDTO(
            id=project.id,
            title=project.title,
            topic=project.topic,
            status=project.status,
            created_by=project.created_by,
            protocol_id=project.protocol_id,
        )
