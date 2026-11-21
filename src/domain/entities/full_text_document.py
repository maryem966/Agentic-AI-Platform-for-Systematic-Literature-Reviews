"""
Entité : FullTextDocument
Suivi du statut et du chemin des PDFs téléchargés.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FullTextDocument(BaseModel):
    """
    Document PDF associé à une publication.
    Tracé avec un hash SHA-256 pour détecter les doublons.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    publication_id: str
    file_path: str                 # Chemin local du PDF (relatif à data/raw/pdfs/)
    file_hash: str                 # SHA-256 — détection des doublons
    extraction_status: Literal[
        "pending",       # PDF téléchargé, extraction non lancée
        "extracted",     # Texte extrait avec succès
        "failed",        # Extraction échouée (PDF corrompu, protégé...)
        "inaccessible",  # PDF non disponible (accès payant, lien mort...)
    ] = "pending"
    extraction_error: str | None = None
    page_count: int | None = None
    extracted_at: datetime | None = None
    extractor_version: str | None = None  # Ex: "pymupdf-1.24.13"

    @property
    def is_available(self) -> bool:
        """True si le texte est extrait et disponible."""
        return self.extraction_status == "extracted"

    @property
    def is_inaccessible(self) -> bool:
        return self.extraction_status == "inaccessible"

    @property
    def needs_extraction(self) -> bool:
        return self.extraction_status == "pending"

    def mark_extracted(
        self, page_count: int, extractor_version: str
    ) -> "FullTextDocument":
        return self.model_copy(
            update={
                "extraction_status": "extracted",
                "page_count": page_count,
                "extracted_at": datetime.utcnow(),
                "extractor_version": extractor_version,
                "extraction_error": None,
            }
        )

    def mark_failed(self, error: str) -> "FullTextDocument":
        return self.model_copy(
            update={
                "extraction_status": "failed",
                "extraction_error": error,
                "extracted_at": datetime.utcnow(),
            }
        )
