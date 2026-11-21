"""
Use Case : DefineProtocol
Crée un ResearchProtocol à partir des critères fournis.
"""
from __future__ import annotations

from ...domain.entities.research_protocol import (
    DateRange,
    ResearchProtocol,
    ResearchQuestion,
)
from ..dto.research_dtos import DefineProtocolDTO


class DefineProtocolUseCase:
    """
    Use case : Définir le protocole de recherche.
    
    Phase 0 : Stub sans persistence.
    Phase 1 : Injecter ProtocolRepository.
    Phase 2 : Brancher sur le Planning Agent LangGraph.
    """

    async def execute(self, dto: DefineProtocolDTO) -> ResearchProtocol:
        """
        Construit un ResearchProtocol et vérifie les postconditions.
        """
        questions = [
            ResearchQuestion(text=rq.text, category=rq.category)
            for rq in dto.research_questions
        ]

        date_range: DateRange | None = None
        if dto.start_year or dto.end_year:
            date_range = DateRange(start=dto.start_year, end=dto.end_year)

        protocol = ResearchProtocol(
            research_id=dto.research_id,
            research_questions=questions,
            inclusion_criteria=dto.inclusion_criteria,
            exclusion_criteria=dto.exclusion_criteria,
            target_databases=dto.target_databases,
            date_range=date_range,
            language_filter=dto.language_filter,
        )

        valid, errors = protocol.validate_postconditions()
        if not valid:
            raise ValueError(
                f"Protocole invalide — postconditions non satisfaites : {errors}"
            )

        # Phase 1 : protocol = await self.protocol_repo.save(protocol)
        return protocol
