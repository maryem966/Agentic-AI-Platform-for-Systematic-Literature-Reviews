"""
Entités : Claim, ClaimEvidenceLink
Affirmations du rapport de synthèse et leurs liens aux preuves.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ACCEPTABLE_CLAIM_STATUSES: frozenset[str] = frozenset(
    {
        "supported",
        "partially_supported",
    }
)


class Claim(BaseModel):
    """
    Affirmation faite dans le rapport de synthèse.

    INVARIANT : Aucun Claim ne peut être dans un rapport final accepté
    sans au moins un ClaimEvidenceLink valide.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=10)
    rq_id: str  # Question de recherche concernée
    report_version_id: str
    section: str  # Section du rapport (ex: "Introduction", "RQ1")
    status: Literal[
        "supported",
        "partially_supported",
        "contradicted",
        "unsupported",
        "pending",
    ] = "pending"
    verification_method: Literal[
        "lexical",
        "semantic",
        "human",
        "combined",
    ] = "semantic"
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_verified(self) -> bool:
        return self.status != "pending"

    @property
    def is_acceptable(self) -> bool:
        """Un claim est acceptable dans un rapport final s'il est supporté."""
        return self.status in ACCEPTABLE_CLAIM_STATUSES

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradicting_evidence_ids) > 0

    @property
    def needs_human_review(self) -> bool:
        """True si le claim est contesté ou non supporté."""
        return self.status in ("contradicted", "unsupported")


class ClaimEvidenceLink(BaseModel):
    """
    Lien N-N entre un Claim et une Evidence.
    Chaque lien précise le type de relation et sa confiance.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str
    evidence_id: str
    link_type: Literal[
        "supports",
        "partially_supports",
        "contradicts",
    ]
    link_confidence: float = Field(..., ge=0.0, le=1.0)
    created_by: Literal["auto", "human"]
