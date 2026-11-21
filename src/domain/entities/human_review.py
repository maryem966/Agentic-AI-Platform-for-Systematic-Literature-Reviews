"""
Entité : HumanReview
Décision humaine à un point d'interruption du workflow LangGraph.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# Types de validation humaine (Section 13.1 du CdC)
BLOCKING_REVIEW_TYPES: frozenset[str] = frozenset(
    {
        "protocol_validation",  # Toujours bloquant
        "quality_escalation",  # Bloquant
        "report_approval",  # Toujours bloquant
    }
)

NON_BLOCKING_REVIEW_TYPES: frozenset[str] = frozenset(
    {
        "screening_arbitration",  # Timeout 24h → exclude par défaut
        "neo4j_conflict",  # Timeout 1h → relation non insérée
        "agent_disagreement",  # Timeout 1h → première décision
        "evidence_validation",  # Non bloquant
    }
)


class HumanReview(BaseModel):
    """
    Décision humaine enregistrée à un point d'interruption.
    Le context_snapshot capture l'état exact des données présentées au validateur.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_id: str
    review_type: Literal[
        "protocol_validation",
        "screening_arbitration",
        "evidence_validation",
        "neo4j_conflict",
        "agent_disagreement",
        "quality_escalation",
        "report_approval",
    ]
    reviewer_id: str
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Données exactes présentées au validateur au moment de la décision
    decision: str
    justification: str = Field(..., min_length=5)
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)
    time_to_decision_seconds: int | None = None

    @property
    def is_blocking(self) -> bool:
        """True si ce type de validation est bloquant pour le workflow."""
        return self.review_type in BLOCKING_REVIEW_TYPES

    @property
    def is_approval(self) -> bool:
        return self.review_type == "report_approval"

    @property
    def is_protocol_validation(self) -> bool:
        return self.review_type == "protocol_validation"
