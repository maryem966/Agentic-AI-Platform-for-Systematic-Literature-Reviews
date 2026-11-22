"""Application boundary for the Evidence Registry and its provenance invariants."""

from __future__ import annotations

from typing import Any, Literal, cast

from ..domain.entities.evidence import Evidence
from ..domain.ports.evidence_port import EvidencePort
from ..infrastructure.persistence.models import ChunkORM, EvidenceORM


class EvidenceRegistry:
    def __init__(self, repository: EvidencePort, session: Any) -> None:
        self.repository = repository
        self.session = session

    async def create(self, evidence: Evidence) -> Evidence:
        chunk = await self.session.get(ChunkORM, evidence.chunk_id)
        if chunk is None:
            raise ValueError("Evidence source chunk does not exist")
        if chunk.publication_id != evidence.publication_id:
            raise ValueError("Evidence publication does not match source chunk")
        if evidence.text not in chunk.text:
            raise ValueError("Evidence text must be an exact passage from its source chunk")
        return await self.repository.save_evidence(evidence)

    async def get(self, evidence_id: str) -> Evidence | None:
        return await self.repository.find_evidence_by_id(evidence_id)

    async def list_for_research(self, research_id: str) -> list[Evidence]:
        return await self.repository.find_evidence_by_research(research_id)

    async def update_status(self, evidence_id: str, status: str, **kwargs: Any) -> Evidence:
        allowed = {
            "extracted",
            "auto_verified",
            "auto_rejected",
            "human_validated",
            "human_rejected",
            "contested",
        }
        if status not in allowed:
            raise ValueError("Invalid evidence verification status")
        current = await self.repository.find_evidence_by_id(evidence_id)
        if current is None:
            raise LookupError(f"Evidence not found: {evidence_id}")
        transitions = {
            "extracted": {
                "auto_verified",
                "auto_rejected",
                "human_validated",
                "human_rejected",
                "contested",
            },
            "auto_verified": {"auto_rejected", "human_validated", "human_rejected", "contested"},
            "auto_rejected": {"human_validated", "human_rejected", "contested"},
            "contested": {"human_validated", "human_rejected"},
            "human_validated": set(),
            "human_rejected": set(),
        }
        if status not in transitions[current.verification_status]:
            raise ValueError(
                f"Invalid evidence transition: {current.verification_status} -> {status}"
            )
        current.advance_to(
            cast(
                Literal[
                    "auto_verified",
                    "auto_rejected",
                    "human_validated",
                    "human_rejected",
                    "contested",
                ],
                status,
            ),
            **kwargs,
        )
        return await self.repository.update_verification_status(evidence_id, status, **kwargs)

    async def delete(self, evidence_id: str) -> bool:
        row = await self.session.get(EvidenceORM, evidence_id)
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True
