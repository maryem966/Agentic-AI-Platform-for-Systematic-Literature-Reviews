"""Idempotent PDF-to-PostgreSQL/Qdrant indexing pipeline."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import httpx
from ollama import AsyncClient

from ..config.settings import settings
from ..domain.entities.chunk import Chunk
from ..domain.entities.full_text_document import FullTextDocument
from ..domain.entities.publication import Publication
from ..domain.ports.vector_store import VectorStore
from ..infrastructure.persistence.repositories import (
    ChunkRepo,
    FullTextDocumentRepo,
    PublicationRepo,
)
from .chunking import DocumentChunker, PageText
from .document_ingestion import PDFIngestion
from .retrieval import SentenceTransformerEmbedder


class OllamaEmbedder:
    """Phase 3 temporary 384-dimensional Ollama embedding model adapter."""

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        self.model = model or settings.embedding_model
        self.client = client or AsyncClient(host=settings.ollama_base_url)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings(model=self.model, prompt=text)
        vector = list(response["embedding"])
        if len(vector) != settings.embedding_vector_size:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match Qdrant size "
                f"{settings.embedding_vector_size}; configure a compatible temporary model"
            )
        return vector


class DocumentIndexingPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: OllamaEmbedder | Any | None = None,
        ingestion: PDFIngestion | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.vector_store = vector_store
        if embedder is not None:
            self.embedder = embedder
        elif settings.embedding_provider == "sentence-transformers":
            self.embedder = SentenceTransformerEmbedder(
                settings.embedding_model,
                revision=settings.embedding_model_revision or None,
            )
        elif settings.embedding_provider == "ollama":
            self.embedder = OllamaEmbedder()
        else:
            raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
        self.ingestion = ingestion or PDFIngestion()
        self.chunker = chunker or DocumentChunker(
            settings.chunk_size_tokens, settings.chunk_overlap_tokens, settings.min_chunk_tokens
        )

    async def index(
        self, session: Any, publication: Publication, url: str, directory: Any
    ) -> list[Chunk]:
        try:
            data = await self.ingestion.download(url)
        except (ValueError, OSError, TimeoutError, httpx.HTTPError) as error:
            await self._record_failure(session, publication, error, inaccessible=True)
            return []
        try:
            extracted = self.ingestion.extract(data)
        except (ValueError, OSError) as error:
            failed = FullTextDocument(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"slr-nexus:pdf:{publication.id}")),
                publication_id=publication.id,
                file_path="",
                file_hash=hashlib.sha256(data).hexdigest(),
                extraction_status="failed",
                extraction_error=str(error)[:1000],
            )
            publication.document_id = (await FullTextDocumentRepo(session).save(failed)).id
            await PublicationRepo(session).save(publication)
            return []
        path = self.ingestion.persist(data, directory, extracted.sha256)
        import fitz

        document = FullTextDocument(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"slr-nexus:pdf:{publication.id}")),
            publication_id=publication.id,
            file_path=str(path),
            file_hash=extracted.sha256,
            extraction_status="extracted",
            page_count=len(extracted.pages),
            extractor_version=f"pymupdf-{fitz.VersionBind}",
        )
        document = await FullTextDocumentRepo(session).save(document)
        publication.document_id = document.id
        await PublicationRepo(session).save(publication)
        source_pages = [PageText(page=number, text=text) for number, text in extracted.pages]
        chunks = self.chunker.split(
            source_pages,
            document_id=document.id,
            publication_id=publication.id,
            embedding_model=self.embedder.model,
            embedding_version=settings.embedding_model_revision or "unfrozen",
        )
        for chunk in chunks:
            chunk.id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"slr-nexus:chunk:{document.id}:{chunk.chunk_index}",
                )
            )
        await self.vector_store.create_collection(
            settings.qdrant_chunks_collection, settings.embedding_vector_size
        )
        chunks = await ChunkRepo(session).save_many(chunks)
        items: list[tuple[str, list[float], dict[str, Any]]] = []
        for chunk in chunks:
            vector = await self.embedder.embed(chunk.text)
            items.append(
                (
                    chunk.id,
                    vector,
                    {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "publication_id": chunk.publication_id,
                        "page": chunk.page,
                        "section_type": chunk.section_type,
                        "text": chunk.text,
                    },
                )
            )
        await self.vector_store.bulk_upsert(settings.qdrant_chunks_collection, items)
        await session.flush()
        return chunks

    @staticmethod
    async def _record_failure(
        session: Any, publication: Publication, error: Exception, *, inaccessible: bool
    ) -> None:
        document = FullTextDocument(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"slr-nexus:pdf:{publication.id}")),
            publication_id=publication.id,
            file_path="",
            file_hash=None,
            extraction_status="inaccessible" if inaccessible else "failed",
            extraction_error=str(error)[:1000],
        )
        publication.document_id = (await FullTextDocumentRepo(session).save(document)).id
        await PublicationRepo(session).save(publication)
