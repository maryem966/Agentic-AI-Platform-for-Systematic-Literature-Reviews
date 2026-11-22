"""Qdrant implementation of the domain VectorStore port."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from ...config.settings import settings
from ...domain.ports.vector_store import VectorSearchResult, VectorStore


class QdrantVectorStoreAdapter(VectorStore):
    """Async Qdrant adapter with deterministic point IDs and collection bootstrap."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        vector_size: int | None = None,
        client: AsyncQdrantClient | Any | None = None,
    ) -> None:
        self.vector_size = vector_size or settings.embedding_vector_size
        self._collection_sizes: dict[str, int] = {}
        if self.vector_size < 1:
            raise ValueError("vector_size must be positive")
        self.client = client or AsyncQdrantClient(
            url=url or f"http://{settings.qdrant_host}:{settings.qdrant_port}",
            api_key=api_key or settings.qdrant_api_key or None,
            timeout=settings.llm_timeout_seconds,
        )

    @staticmethod
    def _point_id(collection: str, document_id: str) -> str:
        # Qdrant accepts UUID or uint64 IDs. UUID5 makes retries naturally idempotent.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slr-nexus:{collection}:{document_id}"))

    @staticmethod
    def _distance(name: str) -> models.Distance:
        try:
            return {
                "cosine": models.Distance.COSINE,
                "dot": models.Distance.DOT,
                "euclid": models.Distance.EUCLID,
            }[name.lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported Qdrant distance: {name}") from error

    async def ensure_collections(self) -> None:
        """Create the two Phase 1 collections if missing."""
        await self.create_collection(settings.qdrant_chunks_collection, self.vector_size)
        await self.create_collection(settings.qdrant_memory_collection, self.vector_size)

    async def collection_exists(self, collection: str) -> bool:
        response = await self.client.get_collections()
        return any(item.name == collection for item in response.collections)

    async def create_collection(
        self, collection: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be positive")
        if await self.collection_exists(collection):
            existing = await self.client.get_collection(collection_name=collection)
            config = existing.config.params.vectors
            existing_size = config.size if hasattr(config, "size") else None
            if existing_size is not None:
                self._collection_sizes[collection] = existing_size
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"Collection {collection!r} has vector size {existing_size}; expected {vector_size}"
                )
            return
        await self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=self._distance(distance)),
        )
        self._collection_sizes[collection] = vector_size

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if not vector or any(
            not isinstance(value, float | int) or isinstance(value, bool) for value in vector
        ):
            raise ValueError("vector must contain numeric values")

    def _validate_vector_size(self, collection: str, vector: list[float]) -> None:
        self._validate_vector(vector)
        expected = self._collection_sizes.get(collection, self.vector_size)
        if len(vector) != expected:
            raise ValueError(
                f"vector has size {len(vector)}; collection {collection!r} expects {expected}"
            )

    async def upsert(
        self, collection: str, doc_id: str, vector: list[float], payload: dict
    ) -> None:
        await self.bulk_upsert(collection, [(doc_id, vector, payload)])

    async def bulk_upsert(
        self, collection: str, items: list[tuple[str, list[float], dict]]
    ) -> None:
        if not items:
            return
        points = []
        for doc_id, vector, payload in items:
            self._validate_vector_size(collection, vector)
            point_payload = dict(payload)
            point_payload["_slr_document_id"] = doc_id
            points.append(
                models.PointStruct(
                    id=self._point_id(collection, doc_id), vector=vector, payload=point_payload
                )
            )
        # Qdrant upsert replaces a point with the same deterministic ID, making retries safe.
        await self.client.upsert(collection_name=collection, points=points, wait=True)

    @staticmethod
    def _filter(filters: dict | None) -> models.Filter | None:
        if not filters:
            return None
        conditions = [
            models.FieldCondition(
                key=key,
                match=(
                    models.MatchAny(any=value)
                    if isinstance(value, list)
                    else models.MatchValue(value=value)
                ),
            )
            for key, value in filters.items()
        ]
        return models.Filter(must=conditions)

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[VectorSearchResult]:
        self._validate_vector_size(collection, query_vector)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        points = await self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=self._filter(filters),
            with_payload=True,
        )
        results = []
        for point in points:
            payload = dict(point.payload or {})
            original_id = payload.pop("_slr_document_id", str(point.id))
            results.append(VectorSearchResult(id=original_id, score=point.score, payload=payload))
        return results

    async def delete(self, collection: str, doc_id: str) -> None:
        await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[self._point_id(collection, doc_id)]),
            wait=True,
        )

    async def count(self, collection: str) -> int:
        response = await self.client.count(collection_name=collection, exact=True)
        return response.count

    async def health_check(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self.client.close()
