"""
Entités : SearchRun, SearchQuery
Traçabilité complète des recherches exécutées sur les APIs scientifiques.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """
    Requête exacte envoyée à une source scientifique.
    La requête est conservée telle quelle — jamais paraphrasée.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    search_run_id: str
    source: Literal["openalex", "crossref", "semantic_scholar", "manual"]
    query_string: str  # Requête exacte envoyée à l'API
    filters_applied: dict[str, Any] = Field(default_factory=dict)  # Date, langue, type...
    results_count: int = 0
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    http_status: int | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """True si la requête a renvoyé un statut HTTP 2xx."""
        return self.http_status is not None and 200 <= self.http_status < 300

    @property
    def failed(self) -> bool:
        return not self.succeeded and self.http_status is not None


class SearchRun(BaseModel):
    """
    Exécution d'une série de recherches pour une version de protocole donnée.
    Contient toutes les requêtes exécutées et les comptages par source.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_id: str
    protocol_id: str  # Version exacte du protocole utilisée
    protocol_version: int
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    queries: list[SearchQuery] = Field(default_factory=list)
    results_by_source: dict[str, int] = Field(default_factory=dict)
    # Ex: {"openalex": 42, "crossref": 18, "semantic_scholar": 25}
    duplicates_removed: int = 0
    agent_execution_id: str | None = None  # Lien vers AgentExecution

    @property
    def total_results(self) -> int:
        """Nombre total de résultats avant déduplication."""
        return sum(self.results_by_source.values())

    @property
    def total_after_dedup(self) -> int:
        """Nombre de publications uniques après déduplication."""
        return max(0, self.total_results - self.duplicates_removed)

    @property
    def has_results(self) -> bool:
        return self.total_results > 0

    @property
    def sources_queried(self) -> list[str]:
        return list(self.results_by_source.keys())
