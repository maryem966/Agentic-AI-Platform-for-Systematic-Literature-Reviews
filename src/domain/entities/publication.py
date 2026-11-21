"""
Entités : Publication, Author
Métadonnées bibliographiques centralisées.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class Author(BaseModel):
    """Auteur d'une publication scientifique."""

    name: str
    orcid: str | None = None
    affiliation: str | None = None


class Publication(BaseModel):
    """
    Métadonnées bibliographiques centralisées.
    Source de vérité pour l'identité d'une publication dans le système.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    external_ids: dict[str, str] = Field(default_factory=dict)
    # Ex: {"doi": "10.1234/...", "openalex_id": "W...", "semantic_scholar_id": "..."}
    title: str
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    url: str | None = None
    source: str                    # Source de découverte : "openalex", "crossref"...
    journal: str | None = None
    document_id: str | None = None # Lien vers FullTextDocument si PDF disponible
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def has_doi(self) -> bool:
        return bool(self.doi)

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract and self.abstract.strip())

    @property
    def has_full_text(self) -> bool:
        return self.document_id is not None

    def get_citation_key(self) -> str:
        """Génère une clé de citation (PremierAuteur + Année)."""
        if self.authors:
            last_name = self.authors[0].name.split()[-1]
        else:
            last_name = "Unknown"
        year = str(self.year) if self.year else "nd"
        return f"{last_name}{year}"

    def get_display_id(self) -> str:
        """Retourne l'identifiant externe préféré pour l'affichage."""
        for key in ("doi", "openalex_id", "semantic_scholar_id"):
            if key in self.external_ids:
                return f"{key}:{self.external_ids[key]}"
        return self.id
