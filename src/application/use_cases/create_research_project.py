"""
Use Case : CreateResearchProject
Crée un nouveau projet SLR.
"""
from __future__ import annotations

from ...domain.entities.research_project import ResearchProject
from ..dto.research_dtos import CreateResearchProjectDTO, ResearchProjectResponseDTO


class CreateResearchProjectUseCase:
    """
    Use case : Créer un nouveau projet SLR.
    
    Phase 0 : Implémentation stub — pas de persistence.
    Phase 1 : Injecter ProjectRepository pour la persistence PostgreSQL.
    """

    # En Phase 1, on injectera : project_repo: ProjectRepository
    def __init__(self) -> None:
        pass

    async def execute(
        self, dto: CreateResearchProjectDTO
    ) -> ResearchProjectResponseDTO:
        """
        Crée un ResearchProject à partir du DTO.
        Validation Pydantic automatique par le modèle de domaine.
        """
        project = ResearchProject(
            title=dto.title,
            topic=dto.topic,
            created_by=dto.created_by,
        )

        # Phase 1 : project = await self.project_repo.save(project)
        # Pour Phase 0, on retourne directement l'entité créée

        return ResearchProjectResponseDTO(
            id=project.id,
            title=project.title,
            topic=project.topic,
            status=project.status,
            created_by=project.created_by,
            protocol_id=project.protocol_id,
        )
