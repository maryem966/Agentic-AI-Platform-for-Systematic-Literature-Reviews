"""
Entité : ScreeningDecision
Décision d'inclusion/exclusion pour une publication à une étape du screening.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Codes de motifs d'exclusion standardisés
STANDARD_EXCLUSION_REASONS: frozenset[str] = frozenset({
    "wrong_population",
    "wrong_intervention",
    "wrong_outcome",
    "wrong_study_design",
    "wrong_language",
    "wrong_date",
    "duplicate",
    "not_peer_reviewed",
    "insufficient_data",
    "full_text_unavailable",
    "retracted",
    "other",
})


class ScreeningDecision(BaseModel):
    """
    Décision de screening par étape avec justification complète.
    Les décisions automatiques sont des recommandations — pas des décisions définitives.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    publication_id: str
    research_id: str
    protocol_id: str
    screening_stage: Literal["title_abstract", "full_text"]
    decision: Literal["include", "exclude", "uncertain"]
    exclusion_reason: str | None = None  # Code standardisé si decision == "exclude"
    justification: str = Field(..., min_length=10)
    criteria_applied: list[str] = Field(default_factory=list)
    agent: Literal["deterministic", "llm"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    human_review_id: str | None = None
    evaluator_2_decision_id: str | None = None  # Second évaluateur (Cohen's Kappa)
    agreement_score: float | None = Field(None, ge=-1.0, le=1.0)  # Cohen's Kappa
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_exclusion_reason(self) -> "ScreeningDecision":
        """Un motif d'exclusion est obligatoire quand decision == 'exclude'."""
        if self.decision == "exclude":
            if not self.exclusion_reason:
                raise ValueError(
                    "exclusion_reason est obligatoire quand decision == 'exclude'"
                )
            if self.exclusion_reason not in STANDARD_EXCLUSION_REASONS:
                raise ValueError(
                    f"Motif d'exclusion non standardisé : '{self.exclusion_reason}'. "
                    f"Valeurs autorisées : {sorted(STANDARD_EXCLUSION_REASONS)}"
                )
        return self

    @property
    def requires_human_arbitration(self) -> bool:
        """
        True si cette décision doit être soumise à arbitrage humain.
        Déclencheur : uncertain ET confidence < 0.6 (Section 13.1 CdC).
        """
        return self.decision == "uncertain" and self.confidence < 0.6

    @property
    def is_human_reviewed(self) -> bool:
        return self.human_review_id is not None

    def override_with_human_decision(
        self,
        new_decision: Literal["include", "exclude"],
        human_review_id: str,
        exclusion_reason: str | None = None,
    ) -> "ScreeningDecision":
        """Remplace la décision automatique par la décision humaine."""
        return self.model_copy(
            update={
                "decision": new_decision,
                "exclusion_reason": exclusion_reason,
                "human_review_id": human_review_id,
                "agent": "llm",  # Préserve l'origine, human_review_id marque l'arbitrage
            }
        )
