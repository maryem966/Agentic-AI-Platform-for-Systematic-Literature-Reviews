from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.infrastructure.vector.qdrant_adapter import QdrantVectorStoreAdapter
from src.rag.evaluation import retrieval_metrics
from src.rag.golden_dataset import GoldenDataset, create_annotation_template
from src.rag.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RetrievalDocument,
    RetrievalHit,
    RetrievalRouter,
)


def _complete(coroutine):
    with pytest.raises(StopIteration) as result:
        coroutine.send(None)
    return result.value.value


def test_bm25_ranks_documents_by_term_match() -> None:
    retriever = BM25Retriever(
        [
            RetrievalDocument("weak", "research methods and survey"),
            RetrievalDocument("strong", "neural networks improve image classification"),
        ]
    )
    hits = _complete(retriever.retrieve("neural networks image", 2))
    assert hits[0].chunk_id == "strong"


def test_rrf_rewards_consistent_rank_and_is_stable() -> None:
    retriever = object.__new__(HybridRetriever)
    retriever.rrf_k = 60
    sparse = [RetrievalHit("both", "shared", 3), RetrievalHit("sparse", "only sparse", 2)]
    dense = [RetrievalHit("both", "shared", 1), RetrievalHit("dense", "only dense", 0.8)]
    fused = retriever._rrf(sparse, dense)
    assert fused[0].chunk_id == "both"
    assert {item.chunk_id for item in fused} == {"both", "sparse", "dense"}


def test_router_uses_transparent_four_level_profiles() -> None:
    router = RetrievalRouter()
    assert router.route("What is RAG?").complexity == "simple"
    assert router.route(
        "Compare longitudinal mechanisms and effects across interventions and populations"
    ).complexity in {"complex", "very_complex"}


def test_retrieval_metrics_use_human_relevance_ids() -> None:
    result = retrieval_metrics({"relevant-a", "relevant-b"}, ["other", "relevant-a", "relevant-b"])
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.mrr == 0.5
    assert 0 < result.ndcg_at_10 <= 1


def test_blank_annotation_template_cannot_be_mistaken_for_gold_data() -> None:
    template = create_annotation_template(25)
    assert len(template["questions"]) == 25
    with pytest.raises(ValidationError):
        GoldenDataset.model_validate(template)


def test_qdrant_filter_supports_research_scoped_publication_sets() -> None:
    qdrant_filter = QdrantVectorStoreAdapter._filter({"publication_id": ["pub-a", "pub-b"]})
    assert qdrant_filter is not None
    assert qdrant_filter.must[0].match.any == ["pub-a", "pub-b"]
