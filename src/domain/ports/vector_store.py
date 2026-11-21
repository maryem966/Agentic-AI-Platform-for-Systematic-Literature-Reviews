"""
Port : VectorStore
Interface abstraite pour le stockage et la recherche vectorielle (implémentée par Qdrant).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorSearchResult:
    """Résultat d'une recherche vectorielle."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Port vers la base de données vectorielle."""

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        doc_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """
        Insère ou met à jour un vecteur dans une collection.
        IDEMPOTENT : si l'ID existe, il est mis à jour.
        """
        ...

    @abstractmethod
    async def bulk_upsert(
        self,
        collection: str,
        items: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Insertion en lot. Idempotent. items = [(id, vector, payload), ...]"""
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Recherche les top_k vecteurs les plus proches."""
        ...

    @abstractmethod
    async def delete(self, collection: str, doc_id: str) -> None:
        """Supprime un vecteur par son ID."""
        ...

    @abstractmethod
    async def collection_exists(self, collection: str) -> bool:
        """Vérifie si une collection existe."""
        ...

    @abstractmethod
    async def create_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Crée une collection si elle n'existe pas."""
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Retourne le nombre de vecteurs dans une collection."""
        ...
