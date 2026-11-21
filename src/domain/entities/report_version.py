"""
Entité : ReportVersion
Version d'un rapport de synthèse SLR, avec ses métadonnées de qualité.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ReportVersion(BaseModel):
    """
    Version versionnée d'un rapport de synthèse.
    Chaque révision crée une nouvelle ReportVersion (jamais de modification en place).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_id: str
    version: int
    content: dict = Field(default_factory=dict)
    # Structure : {"introduction": "...", "rq1": "...", "rq2": "...", "conclusion": "..."}
    generated_by_agent: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    overall_quality_score: float | None = Field(None, ge=0.0, le=1.0)
    human_approved: bool = False
    human_review_id: str | None = None
    previous_version_id: str | None = None  # Lien vers la version précédente

    @property
    def is_final(self) -> bool:
        """Un rapport est final s'il a été approuvé par un humain."""
        return self.human_approved

    @property
    def is_first_version(self) -> bool:
        return self.previous_version_id is None

    def get_section(self, section_name: str) -> str | None:
        """Retourne le contenu d'une section par son nom."""
        return self.content.get(section_name)

    def get_all_sections(self) -> list[str]:
        """Retourne les noms de toutes les sections disponibles."""
        return list(self.content.keys())

    def mark_approved(self, human_review_id: str) -> "ReportVersion":
        """Approuve le rapport — action humaine obligatoire."""
        return self.model_copy(
            update={
                "human_approved": True,
                "human_review_id": human_review_id,
            }
        )
