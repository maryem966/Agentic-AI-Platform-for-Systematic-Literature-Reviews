"""
Adaptateur Qdrant pour la recherche vectorielle (VectorStore).
Stocke les embeddings des chunks pour le RAG de la Phase 3/4.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest

from ...config.settings import settings
from ...domain.ports.vector_store import VectorSearchResult, VectorStore


class QdrantVectorStoreAdapter(VectorStore):
    """Implémentation du port VectorStore avec Qdrant."""

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        if client:
            self.client = client
        else:
            self.client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key or None,
            )

    async def collection_exists(self, collection: str) -> bool:
        """Vérifie si une collection existe déjà."""
        try:
            collections = await self.client.get_collections()
            return any(c.name == collection for c in collections.collections)
        except Exception:
            return False

    async def create_collection(
        self,
        collection: str,
        vector_size: int = 384,  # Taille standard pour all-MiniLM-L6-v2
        distance: str = "Cosine",
    ) -> None:
        """Crée une collection de vecteurs si elle n'existe pas encore."""
        dist = rest.Distance.COSINE
        if distance.lower() == "euclid":
            dist = rest.Distance.EUCLID
        elif distance.lower() == "dot":
            dist = rest.Distance.DOT

        exists = await self.collection_exists(collection)
        if not exists:
            await self.client.create_collection(
                collection_name=collection,
                vectors_config=rest.VectorParams(size=vector_size, distance=dist),
            )

    async def upsert(
        self,
        collection: str,
        doc_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insère ou met à jour un point (idempotent par doc_id)."""
        await self.client.upsert(
            collection_name=collection,
            points=[
                rest.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    async def bulk_upsert(
        self,
        collection: str,
        items: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Insertion par lot pour optimiser les performances d'indexation."""
        if not items:
            return

        points = [
            rest.PointStruct(id=doc_id, vector=vector, payload=payload)
            for doc_id, vector, payload in items
        ]
        await self.client.upsert(collection_name=collection, points=points)

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Recherche par similarité cosinus avec filtres optionnels."""
        query_filter = None
        if filters:
            conditions = []
            for key, val in filters.items():
                conditions.append(
                    rest.FieldCondition(
                        key=key,
                        match=rest.MatchValue(value=val),
                    )
                )
            query_filter = rest.Filter(must=conditions)

        results = await self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            VectorSearchResult(
                id=str(r.id),
                score=float(r.score),
                payload=dict(r.payload or {}),
            )
            for r in results
        ]

    async def delete(self, collection: str, doc_id: str) -> None:
        """Supprime un vecteur par son identifiant."""
        await self.client.delete(
            collection_name=collection,
            points_selector=rest.PointIdsList(points=[doc_id]),
        )

    async def count(self, collection: str) -> int:
        """Compte le nombre total de vecteurs indexés."""
        try:
            info = await self.client.get_collection(collection_name=collection)
            return info.points_count or 0
        except Exception:
            return 0
