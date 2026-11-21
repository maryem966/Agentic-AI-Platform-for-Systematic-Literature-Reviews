"""
Service : ClaimVerifier
Vérification des claims en 4 niveaux (Section 11.1 du CdC).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from ..entities.claim import Claim
from ..entities.evidence import Evidence


@dataclass
class ClaimVerification:
    """Résultat de la vérification d'un Claim."""

    claim_id: str
    status: Literal[
        "supported",
        "partially_supported",
        "contradicted",
        "unsupported",
    ]
    reason: str | None
    verification_method: Literal[
        "existence",   # Niveau 1
        "provenance",  # Niveau 2
        "semantic",    # Niveau 3
        "entailment",  # Niveau 4
        "combined",    # Plusieurs niveaux
    ]
    semantic_similarity_score: float | None = None
    entailment_label: str | None = None          # "entailment", "neutral", "contradiction"
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence_ids: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    verified_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_acceptable(self) -> bool:
        return self.status in ("supported", "partially_supported")


class ClaimVerifier(ABC):
    """
    Service de vérification des claims en 4 niveaux.

    Niveau 1 — Existence : les IDs d'evidence existent dans le registre
    Niveau 2 — Provenance : le chunk appartient bien à la publication citée
    Niveau 3 — Pertinence sémantique : similarité embedding claim ↔ passage
    Niveau 4 — Entailment : le passage supporte-t-il ou contredit-il le claim ?

    Note : Le chevauchement lexical de tokens N'EST PAS un verdict scientifique.
    """

    @abstractmethod
    async def verify(
        self,
        claim: Claim,
        evidences: list[Evidence],
    ) -> ClaimVerification:
        """
        Vérifie un claim en 4 niveaux.
        S'arrête au premier niveau qui échoue.
        """
        ...

    @abstractmethod
    async def find_contradicting_evidence(
        self,
        claim: Claim,
        registry: list[Evidence],
    ) -> list[Evidence]:
        """
        Cherche des preuves contradictoires dans d'autres publications.
        Utilisé avant l'acceptation du rapport final.
        """
        ...
