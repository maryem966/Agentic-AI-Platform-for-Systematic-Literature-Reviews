"""
Entité : Evidence
Preuve vérifiable issue d'une publication — passage source EXACT (jamais paraphrasé).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# États du cycle de vie d'une preuve
TRUSTWORTHY_STATUSES: frozenset[str] = frozenset(
    {
        "auto_verified",
        "human_validated",
    }
)

REJECTED_STATUSES: frozenset[str] = frozenset(
    {
        "auto_rejected",
        "human_rejected",
    }
)


class Evidence(BaseModel):
    """
    Preuve scientifique tracée jusqu'à son passage source exact.

    INVARIANT : text est le passage source EXACT extrait du PDF.
    Il ne doit JAMAIS être paraphrasé, résumé ou modifié.

    verification_status remplace validated: bool (v4.0 → v5.0).
    Les 6 états couvrent le cycle de vie complet d'une preuve.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    publication_id: str
    chunk_id: str  # Lien vers le Chunk source
    text: str = Field(..., min_length=1)  # Passage source EXACT
    page: int | None = None
    section: str | None = None
    evidence_type: Literal[
        "result",
        "method",
        "limitation",
        "dataset",
        "metric",
        "definition",
    ]
    extracted_by_agent: str  # Nom de l'agent ayant extrait la preuve
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    verification_status: Literal[
        "extracted",  # Brute — pas encore vérifiée
        "auto_verified",  # Vérification automatique passée (4 niveaux)
        "auto_rejected",  # Vérification automatique échouée
        "human_validated",  # Validée explicitement par un humain
        "human_rejected",  # Rejetée par un humain
        "contested",  # Contestée — arbitrage en attente
    ] = "extracted"
    verification_method: str | None = None  # Description de la méthode utilisée
    human_review_id: str | None = None
    research_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_trustworthy(self) -> bool:
        """Une preuve fiable est vérifiée automatiquement ou validée humainement."""
        return self.verification_status in TRUSTWORTHY_STATUSES

    @property
    def is_rejected(self) -> bool:
        return self.verification_status in REJECTED_STATUSES

    @property
    def requires_human_review(self) -> bool:
        """True si la preuve est rejetée automatiquement ou contestée."""
        return self.verification_status in ("auto_rejected", "contested")

    @property
    def is_pending(self) -> bool:
        return self.verification_status == "extracted"

    def advance_to(
        self,
        new_status: Literal[
            "auto_verified", "auto_rejected", "human_validated", "human_rejected", "contested"
        ],
        verification_method: str | None = None,
        human_review_id: str | None = None,
    ) -> Evidence:
        """Fait avancer la preuve dans son cycle de vie."""
        return self.model_copy(
            update={
                "verification_status": new_status,
                "verification_method": verification_method or self.verification_method,
                "human_review_id": human_review_id or self.human_review_id,
            }
        )
