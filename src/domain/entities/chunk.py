"""
Entité : Chunk
Passage extrait d'un document, prêt pour l'indexation vectorielle dans Qdrant.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Passage de texte extrait d'un document PDF.
    Les métadonnées sont dans PostgreSQL ; l'embedding est dans Qdrant.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str  # FK vers FullTextDocument
    publication_id: str  # FK vers Publication (pour lookup rapide)
    text: str  # Texte brut du chunk
    page: int | None = None
    section: str | None = None  # Ex: "Introduction", "Methods", "Results"
    section_type: Literal[
        "text",  # Texte narratif standard
        "table",  # Tableau de données
        "figure",  # Légende de figure
        "formula",  # Équation ou formule mathématique
        "reference",  # Référence bibliographique
    ] = "text"
    chunk_index: int  # Position dans le document
    char_start: int  # Offset de début dans le texte du document
    char_end: int  # Offset de fin dans le texte du document
    embedding_model: str  # Ex: "sentence-transformers/all-MiniLM-L6-v2"
    embedding_version: str  # Version du modèle — pour invalidation si changement
    indexed_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def length(self) -> int:
        """Longueur en caractères du chunk."""
        return self.char_end - self.char_start

    @property
    def token_count_estimate(self) -> int:
        """Estimation grossière du nombre de tokens (1 token ≈ 4 chars)."""
        return max(1, self.length // 4)

    @property
    def is_too_short(self) -> bool:
        """
        True si le chunk est trop court (< 100 tokens estimés).
        Les chunks trop courts doivent être fusionnés (Section 10.2 CdC).
        """
        return self.token_count_estimate < 100

    @property
    def is_semantic_reranking_eligible(self) -> bool:
        """
        Les tableaux et formules sont exclus du reranking sémantique
        (Section 10.2 CdC) — ils contiennent des données structurées,
        pas du texte narratif comparable à une query.
        """
        return self.section_type not in ("table", "formula")
