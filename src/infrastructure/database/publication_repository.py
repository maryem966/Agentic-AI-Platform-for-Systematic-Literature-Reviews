"""
Adaptateur PostgreSQL pour PublicationRepository.
Fait la passerelle entre les entités métier (Publication) et les tables SQLAlchemy.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.publication import Author, Publication
from ...domain.ports.publication_repository import PublicationRepository
from .models import PublicationModel, ScreeningDecisionModel


class PostgresPublicationRepository(PublicationRepository):
    """Implémentation PostgreSQL du port PublicationRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _to_entity(self, model: PublicationModel) -> Publication:
        """Convertit un modèle SQLAlchemy en entité pure du domaine."""
        authors = [
            Author(
                name=a.get("name", ""),
                orcid=a.get("orcid"),
                affiliation=a.get("affiliation"),
            )
            for a in (model.authors or [])
        ]
        return Publication(
            id=model.id,
            external_ids=model.external_ids or {},
            title=model.title,
            authors=authors,
            year=model.year,
            abstract=model.abstract,
            doi=model.doi,
            url=model.url,
            source=model.source,
            journal=model.journal,
            document_id=model.document_id,
            created_at=model.created_at,
        )

    def _to_model(self, entity: Publication, research_id: str) -> PublicationModel:
        """Convertit une entité du domaine en modèle SQLAlchemy."""
        authors_json = [
            {"name": a.name, "orcid": a.orcid, "affiliation": a.affiliation}
            for a in entity.authors
        ]
        return PublicationModel(
            id=entity.id,
            research_id=research_id,
            external_ids=entity.external_ids,
            title=entity.title,
            authors=authors_json,
            year=entity.year,
            abstract=entity.abstract,
            doi=entity.doi,
            url=entity.url,
            source=entity.source,
            journal=entity.journal,
            document_id=entity.document_id,
            created_at=entity.created_at,
        )

    async def save(self, publication: Publication, research_id: str = "default") -> Publication:
        """Persiste ou met à jour une publication (idempotent)."""
        existing = await self.session.get(PublicationModel, publication.id)
        if existing:
            existing.title = publication.title
            existing.abstract = publication.abstract
            existing.doi = publication.doi
            existing.url = publication.url
            existing.document_id = publication.document_id
        else:
            model = self._to_model(publication, research_id)
            self.session.add(model)
        await self.session.flush()
        return publication

    async def find_by_id(self, publication_id: str) -> Publication | None:
        """Trouve une publication par son identifiant unique."""
        result = await self.session.get(PublicationModel, publication_id)
        return self._to_entity(result) if result else None

    async def find_by_doi(self, doi: str) -> Publication | None:
        """Recherche par DOI pour la déduplication."""
        stmt = select(PublicationModel).where(PublicationModel.doi == doi).limit(1)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_external_id(self, source: str, external_id: str) -> Publication | None:
        """Recherche par ID externe (OpenAlex, Crossref)."""
        # Recherche dans le JSONB external_ids
        stmt = select(PublicationModel).where(
            PublicationModel.external_ids[source].as_string() == external_id
        ).limit(1)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_research_id(self, research_id: str) -> list[Publication]:
        """Retourne toutes les publications associées à une recherche."""
        stmt = select(PublicationModel).where(PublicationModel.research_id == research_id)
        result = await self.session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_included_by_research_id(self, research_id: str) -> list[Publication]:
        """Retourne uniquement les publications ayant une décision 'include' au screening."""
        stmt = (
            select(PublicationModel)
            .join(ScreeningDecisionModel, PublicationModel.id == ScreeningDecisionModel.publication_id)
            .where(
                PublicationModel.research_id == research_id,
                ScreeningDecisionModel.decision == "include",
            )
            .distinct()
        )
        result = await self.session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def count_by_research_id(self, research_id: str) -> int:
        """Compte le nombre de publications associées à une recherche."""
        stmt = (
            select(func.count())
            .select_from(PublicationModel)
            .where(PublicationModel.research_id == research_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def bulk_save(self, publications: list[Publication], research_id: str = "default") -> list[Publication]:
        """Sauvegarde en lot un groupe de publications (avec déduplication)."""
        saved: list[Publication] = []
        for pub in publications:
            # Vérification de doublon par DOI si présent
            if pub.doi:
                existing = await self.find_by_doi(pub.doi)
                if existing:
                    saved.append(existing)
                    continue
            s = await self.save(pub, research_id)
            saved.append(s)
        return saved
