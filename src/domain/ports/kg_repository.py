"""
Port : KGRepository
Interface abstraite vers le graphe de connaissances Neo4j (Phase 6+).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphResult:
    """Résultat d'une requête sur le graphe de connaissances."""

    data: dict[str, Any] = field(default_factory=dict)
    source_node_id: str | None = None
    relationship_type: str | None = None


class KGRepository(ABC):
    """
    Port vers le Knowledge Graph (Neo4j).
    SÉCURITÉ : Toutes les requêtes passent par des templates Cypher paramétrés.
    Le LLM ne génère JAMAIS de Cypher directement (Section 10.4 CdC).
    """

    @abstractmethod
    async def query_template(
        self,
        template_name: str,
        params: dict[str, Any],
    ) -> list[GraphResult]:
        """
        Exécute un template Cypher pré-approuvé avec des paramètres.
        Lève ValueError si template_name n'est pas dans ALLOWED_TEMPLATE_NAMES.
        Connexion en READ ONLY uniquement.
        """
        ...

    @abstractmethod
    async def node_exists(
        self,
        node_type: str,
        properties: dict[str, Any],
    ) -> bool:
        """Vérifie si un nœud existe dans le graphe."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Vérifie que Neo4j est disponible et accessible."""
        ...
