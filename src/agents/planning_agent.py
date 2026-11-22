"""Planning agent with structured output and explicit postcondition checks."""

from __future__ import annotations

from typing import cast

from ..application.dto.research_dtos import PlanningInput
from ..domain.entities.research_protocol import ResearchProtocol
from ..domain.ports.llm_port import LLMPort


class PlanningAgent:
    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm

    async def plan(self, request: PlanningInput) -> ResearchProtocol:
        prompt = (
            "À partir du sujet et des objectifs ci-dessous, propose un protocole de revue "
            "systématique en français. Les questions doivent être précises; les critères "
            "doivent être vérifiables. Respecte exactement le schéma Pydantic fourni.\n"
            f"Sujet: {request.topic}\nObjectifs: {request.objectives}"
        )
        protocol = cast(
            ResearchProtocol,
            await self.llm.invoke_structured(
                [{"role": "user", "content": prompt}], ResearchProtocol
            ),
        )
        # Validate first, authorize only a protocol for this requested research, then return it.
        protocol = ResearchProtocol.model_validate(protocol)
        if protocol.research_id != request.research_id:
            protocol = protocol.model_copy(update={"research_id": request.research_id})
        valid, errors = protocol.validate_postconditions()
        if not valid:
            raise ValueError(f"Planning refusé : postconditions invalides: {errors}")
        return protocol
