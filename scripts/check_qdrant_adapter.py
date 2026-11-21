"""Create Phase 1 collections and run a disposable Qdrant round-trip check."""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import settings
from src.infrastructure.vector.qdrant_adapter import QdrantVectorStoreAdapter


async def main() -> None:
    store = QdrantVectorStoreAdapter()
    collection = settings.qdrant_chunks_collection
    document_id = f"phase1-smoke-{uuid.uuid4()}"
    vector = [1.0, *([0.0] * (settings.embedding_vector_size - 1))]
    try:
        if not await store.health_check():
            raise RuntimeError("Qdrant health check failed")
        await store.ensure_collections()
        await store.upsert(collection, document_id, vector, {"_smoke": True})
        matches = await store.search(collection, vector, top_k=1, filters={"_smoke": True})
        if not matches or matches[0].id != document_id:
            raise RuntimeError("Qdrant did not return the smoke-test point")
        await store.delete(collection, document_id)
        print("Qdrant OK: collections created; upsert/search/delete round-trip passed.")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
