"""BM25, dense retrieval, reciprocal-rank fusion and optional cross-encoder reranking."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from ..domain.ports.vector_store import VectorStore

RetrievalStrategy = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
Complexity = Literal["simple", "medium", "complex", "very_complex"]


@dataclass(frozen=True)
class RetrievalDocument:
    chunk_id: str
    text: str
    section_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    text: str
    score: float
    section_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


class Embedder(Protocol):
    model: str

    def embed(self, text: str) -> Any: ...


class TextReranker(Protocol):
    model: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class DenseSearch(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]: ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class BM25Retriever:
    """BM25 Okapi with pinned k1/b defaults and no external tokenizer state."""

    def __init__(self, documents: list[RetrievalDocument], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters require k1 > 0 and 0 <= b <= 1")
        self.documents = documents
        self.k1, self.b = k1, b
        self.tokens = [tokenize(document.text) for document in documents]
        self.lengths = [len(document_tokens) for document_tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: dict[str, int] = {}
        for document_tokens in self.tokens:
            for term in set(document_tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        corpus_size = len(self.documents)
        self.idf = {
            term: math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            # The deterministic implementation below keeps unit tests runnable before extras install.
            self._rank_bm25 = None
        else:
            self._rank_bm25 = BM25Okapi(self.tokens, k1=k1, b=b)

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_terms = set(tokenize(query))
        if self._rank_bm25 is not None:
            scores = self._rank_bm25.get_scores(sorted(query_terms)).tolist()
        else:
            scores = []
            for index, terms in enumerate(self.tokens):
                frequencies: dict[str, int] = {}
                for term in terms:
                    if term in query_terms:
                        frequencies[term] = frequencies.get(term, 0) + 1
                score = 0.0
                for term, frequency in frequencies.items():
                    norm = frequency + self.k1 * (
                        1 - self.b + self.b * self.lengths[index] / (self.average_length or 1)
                    )
                    score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / norm
                scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [
            RetrievalHit(
                self.documents[i].chunk_id,
                self.documents[i].text,
                scores[i],
                self.documents[i].section_type,
                self.documents[i].metadata,
            )
            for i in order
        ]


class SentenceTransformerEmbedder:
    """Optional bi-encoder adapter; model loading is deferred until Phase 4 benchmarking."""

    def __init__(self, model: str, device: str | None = None, revision: str | None = None) -> None:
        if not model.strip():
            raise ValueError("Provide a concrete embedding model name or local path")
        self.model = model
        self.device, self.revision = device, revision
        self._encoder: Any | None = None

    def _load(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Install requirements.txt to use Sentence Transformers"
                ) from error
            self._encoder = SentenceTransformer(
                self.model, device=self.device, revision=self.revision
            )
        return self._encoder

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return cast(list[list[float]], vectors.tolist())

    async def embed(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self.encode_many, [text]))[0]


class DenseRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        collection: str,
        filters: dict[str, Any] | None = None,
    ):
        self.vector_store, self.embedder, self.collection = vector_store, embedder, collection
        self.filters = filters

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        vector = self.embedder.embed(query)
        if inspect.isawaitable(vector):
            vector = await vector
        results = await self.vector_store.search(
            self.collection, vector, top_k, filters=self.filters
        )
        return [
            RetrievalHit(
                chunk_id=result.payload.get("chunk_id", result.id),
                text=str(result.payload.get("text", "")),
                score=result.score,
                section_type=str(result.payload.get("section_type", "text")),
                metadata=result.payload,
            )
            for result in results
        ]


class InMemoryDenseRetriever:
    """Dense cosine baseline for the isolated model-selection benchmark."""

    def __init__(
        self,
        documents: list[RetrievalDocument],
        vectors: list[list[float]],
        embedder: Embedder,
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError("Each dense vector must correspond to one corpus document")
        self.documents, self.vectors, self.embedder = documents, vectors, embedder

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        vector = self.embedder.embed(query)
        if inspect.isawaitable(vector):
            vector = await vector
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (
                -_vector_cosine(vector, self.vectors[index]),
                self.documents[index].chunk_id,
            ),
        )[:top_k]
        return [
            RetrievalHit(
                self.documents[index].chunk_id,
                self.documents[index].text,
                _vector_cosine(vector, self.vectors[index]),
                self.documents[index].section_type,
                self.documents[index].metadata,
            )
            for index in order
        ]


def _vector_cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class SentenceTransformerReranker:
    """Cross-encoder scores query/passage pairs; runtime and model must be benchmarked."""

    def __init__(
        self,
        model: str,
        device: str | None = None,
        batch_size: int = 32,
        revision: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Provide a concrete cross-encoder model name or local path")
        self.model, self.device, self.batch_size, self.revision = (
            model,
            device,
            batch_size,
            revision,
        )
        self._encoder: Any | None = None

    def _load(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError("Install requirements.txt to use CrossEncoder") from error
            self._encoder = CrossEncoder(self.model, device=self.device, revision=self.revision)
        return self._encoder

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._load().predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


@dataclass(frozen=True)
class RouteDecision:
    complexity: Complexity
    candidate_k: int
    rerank: bool


class RetrievalRouter:
    """Transparent initial complexity heuristics; final thresholds are benchmark-tunable."""

    COMPLEX_CUES = {
        "compare",
        "comparison",
        "relationship",
        "across",
        "effect",
        "impact",
        "versus",
        "trend",
        "longitudinal",
        "mechanism",
        "why",
        "how does",
        "under what",
    }

    def route(self, query: str) -> RouteDecision:
        terms = tokenize(query)
        normalized = query.casefold()
        cue_count = sum(cue in normalized for cue in self.COMPLEX_CUES)
        conjunctions = len(re.findall(r"\b(and|or|versus|compared with)\b", normalized))
        if len(terms) > 35 or cue_count >= 3 or (cue_count >= 2 and conjunctions >= 2):
            return RouteDecision("very_complex", 30, True)
        if len(terms) > 18 or cue_count >= 2 or conjunctions >= 2:
            return RouteDecision("complex", 20, True)
        if len(terms) > 7 or cue_count or conjunctions:
            return RouteDecision("medium", 10, False)
        return RouteDecision("simple", 5, False)


class HybridRetriever:
    def __init__(
        self,
        sparse: BM25Retriever,
        dense: DenseSearch,
        reranker: TextReranker | None = None,
        rrf_k: int = 60,
        rerank_candidate_k: int = 30,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if rerank_candidate_k < 1:
            raise ValueError("rerank_candidate_k must be positive")
        self.sparse, self.dense, self.reranker, self.rrf_k = sparse, dense, reranker, rrf_k
        self.rerank_candidate_k = rerank_candidate_k
        self.router = RetrievalRouter()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        strategy: RetrievalStrategy = "hybrid",
        *,
        route: bool = False,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if strategy not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("Unsupported retrieval strategy")
        decision = self.router.route(query) if route else None
        candidate_k = max(top_k, decision.candidate_k if decision else top_k)
        if strategy == "hybrid_rerank":
            candidate_k = max(candidate_k, self.rerank_candidate_k)
        if strategy == "bm25":
            hits = await self.sparse.retrieve(query, candidate_k)
        elif strategy == "dense":
            hits = await self.dense.retrieve(query, candidate_k)
        else:
            sparse_hits, dense_hits = await asyncio.gather(
                self.sparse.retrieve(query, candidate_k),
                self.dense.retrieve(query, candidate_k),
            )
            hits = self._rrf(sparse_hits, dense_hits)
        should_rerank = strategy == "hybrid_rerank" or (route and decision and decision.rerank)
        if should_rerank:
            if self.reranker is None:
                raise ValueError(
                    "A reranker is required for hybrid_rerank or routed complex queries"
                )
            hits = await self._rerank(query, hits[:candidate_k])
        return hits[:top_k]

    def _rrf(self, *rankings: list[RetrievalHit]) -> list[RetrievalHit]:
        hits_by_id: dict[str, RetrievalHit] = {}
        fused_scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                hits_by_id.setdefault(hit.chunk_id, hit)
                fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1 / (
                    self.rrf_k + rank
                )
        return [
            RetrievalHit(
                hit.chunk_id, hit.text, fused_scores[chunk_id], hit.section_type, hit.metadata
            )
            for chunk_id, hit in sorted(
                hits_by_id.items(), key=lambda item: (-fused_scores[item[0]], item[0])
            )
        ]

    async def _rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        eligible = [hit for hit in hits if hit.section_type not in {"table", "formula"}]
        structural = [hit for hit in hits if hit.section_type in {"table", "formula"}]
        assert self.reranker is not None
        scores = await asyncio.to_thread(self.reranker.score, query, [hit.text for hit in eligible])
        reranked = [
            RetrievalHit(hit.chunk_id, hit.text, score, hit.section_type, hit.metadata)
            for hit, score in zip(eligible, scores, strict=True)
        ]
        return sorted(reranked, key=lambda hit: (-hit.score, hit.chunk_id)) + structural
