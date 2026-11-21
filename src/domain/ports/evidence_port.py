"""
Port : EvidencePort
Interface abstraite pour l'Evidence Registry (implémentée par PostgreSQL en Phase 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..entities.claim import Claim, ClaimEvidenceLink
from ..entities.evidence import Evidence


class EvidencePort(ABC):
    """Port vers l'Evidence Registry."""

    # --- Evidence ---
    @abstractmethod
    async def save_evidence(self, evidence: Evidence) -> Evidence:
        """Persiste une preuve. Idempotent par chunk_id + evidence_type."""
        ...

    @abstractmethod
    async def find_evidence_by_id(self, evidence_id: str) -> Evidence | None: ...

    @abstractmethod
    async def find_evidence_by_research(self, research_id: str) -> list[Evidence]:
        """Toutes les preuves d'une recherche."""
        ...

    @abstractmethod
    async def find_evidence_by_publication(
        self, publication_id: str, research_id: str
    ) -> list[Evidence]: ...

    @abstractmethod
    async def find_trustworthy_evidence(self, research_id: str) -> list[Evidence]:
        """Preuves auto_verified ou human_validated uniquement."""
        ...

    @abstractmethod
    async def update_verification_status(
        self,
        evidence_id: str,
        new_status: str,
        verification_method: str | None = None,
        human_review_id: str | None = None,
    ) -> Evidence: ...

    # --- Claim ---
    @abstractmethod
    async def save_claim(self, claim: Claim) -> Claim: ...

    @abstractmethod
    async def find_claim_by_id(self, claim_id: str) -> Claim | None: ...

    @abstractmethod
    async def find_claims_by_report(self, report_version_id: str) -> list[Claim]: ...

    # --- ClaimEvidenceLink ---
    @abstractmethod
    async def link_claim_to_evidence(self, link: ClaimEvidenceLink) -> ClaimEvidenceLink:
        """Crée un lien entre un Claim et une Evidence."""
        ...

    @abstractmethod
    async def get_evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        """Retourne toutes les preuves liées à un claim."""
        ...

    @abstractmethod
    async def get_claims_without_evidence(self, report_version_id: str) -> list[Claim]:
        """
        Retourne les claims sans aucune preuve liée.
        Ces claims violent l'invariant : zéro claim sans evidence_id validé.
        """
        ...
