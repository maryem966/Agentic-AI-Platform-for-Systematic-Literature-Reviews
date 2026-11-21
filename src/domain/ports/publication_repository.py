"""
Port : PublicationRepository
Interface abstraite pour la persistance des publications (implémentée en Phase 1 par PostgreSQL).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..entities.publication import Publication


class PublicationRepository(ABC):
    """Port vers le stockage des publications scientifiques."""

    @abstractmethod
    async def save(self, publication: Publication) -> Publication:
        """Persiste ou met à jour une publication. Idempotent par DOI."""
        ...

    @abstractmethod
    async def find_by_id(self, publication_id: str) -> Publication | None:
        """Retourne une publication par son ID interne."""
        ...

    @abstractmethod
    async def find_by_doi(self, doi: str) -> Publication | None:
        """Retourne une publication par son DOI (déduplication)."""
        ...

    @abstractmethod
    async def find_by_external_id(self, source: str, external_id: str) -> Publication | None:
        """Retourne une publication par un ID externe (OpenAlex, Crossref...)."""
        ...

    @abstractmethod
    async def find_by_research_id(self, research_id: str) -> list[Publication]:
        """Retourne toutes les publications d'une recherche."""
        ...

    @abstractmethod
    async def find_included_by_research_id(self, research_id: str) -> list[Publication]:
        """Retourne les publications incluses après screening."""
        ...

    @abstractmethod
    async def count_by_research_id(self, research_id: str) -> int:
        """Compte les publications d'une recherche."""
        ...

    @abstractmethod
    async def bulk_save(self, publications: list[Publication]) -> list[Publication]:
        """Persiste plusieurs publications en une opération. Idempotent."""
        ...
