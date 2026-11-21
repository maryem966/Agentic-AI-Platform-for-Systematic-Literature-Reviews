"""Value Object : MemoryPolicy — Politiques d'isolation de la mémoire"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class MemoryPolicy(BaseModel):
    """
    Politique d'isolation et de rétention de la mémoire sémantique.

    Par défaut : isolation stricte par research_id.
    Le partage inter-recherches est EXPLICITEMENT autorisé par l'utilisateur.

    Section 9.2 du CdC :
    - "research" : mémoire isolée par recherche — pas de contamination inter-projets
    - "user"     : partageable entre recherches du même utilisateur
    - "system"   : partageable globalement
    """

    scope: Literal["research", "user", "system"] = "research"
    retention_days: int | None = None   # None = conservation indéfinie
    allow_cross_research: bool = False  # Par défaut : isolation stricte
    consolidation_rule: str | None = None  # Condition de mémorisation à long terme
    provenance_required: bool = True    # Chaque fait mémorisé a un research_id source

    @property
    def is_isolated(self) -> bool:
        """True si la mémoire est isolée à une recherche spécifique."""
        return self.scope == "research" and not self.allow_cross_research
