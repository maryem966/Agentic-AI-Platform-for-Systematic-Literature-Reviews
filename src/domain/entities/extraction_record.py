"""
Entité : ExtractionRecord
Données structurées extraites des publications par le Research Assistant Agent.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

# Champs standards attendus dans une extraction scientifique
STANDARD_EXTRACTION_FIELDS: frozenset[str] = frozenset({
    "method",
    "dataset",
    "metric",
    "result",
    "limitation",
})


class ExtractionRecord(BaseModel):
    """
    Résultat d'extraction structurée depuis une publication.
    Chaque champ extrait est ancré dans des source_chunk_ids traçables.

    Exemple:
        ExtractionRecord(
            publication_id="P001",
            extracted_fields={
                "method": "Random Forest",
                "dataset": "Credit Card Fraud Dataset",
                "metric": "F1-score",
                "result": "0.91",
                "limitation": "Dataset déséquilibré",
            },
            source_chunk_ids=["C042", "C043"],
        )
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    publication_id: str
    research_id: str
    agent_execution_id: str
    extracted_fields: dict = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)
    # Les chunks d'où proviennent les données extraites
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    validated: bool = False
    human_review_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_complete(self) -> bool:
        """True si tous les champs standards sont remplis."""
        return STANDARD_EXTRACTION_FIELDS.issubset(self.extracted_fields.keys())

    @property
    def missing_fields(self) -> set[str]:
        """Retourne les champs standards manquants."""
        return STANDARD_EXTRACTION_FIELDS - set(self.extracted_fields.keys())

    @property
    def is_anchored(self) -> bool:
        """True si au moins un chunk source est référencé."""
        return len(self.source_chunk_ids) > 0

    def get_field(self, field_name: str) -> str | None:
        return self.extracted_fields.get(field_name)
