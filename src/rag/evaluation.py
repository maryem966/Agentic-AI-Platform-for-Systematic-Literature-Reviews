"""Retrieval, embedding and RAGAS evaluation utilities for the Phase 4 benchmark."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from .golden_dataset import GoldenDataset, GoldenQuestion
from .retrieval import (
    RetrievalDocument,
    RetrievalHit,
    SentenceTransformerEmbedder,
    SentenceTransformerReranker,
)


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


def retrieval_metrics(relevant_ids: set[str], ranked_ids: list[str]) -> RetrievalMetrics:
    if not relevant_ids:
        raise ValueError(
            "Each evaluation question needs at least one human-annotated relevant chunk"
        )
    ranks = [rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevant_ids]

    def recall(k: int) -> float:
        return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)

    reciprocal_rank = 1 / ranks[0] if ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    ideal_count = min(len(relevant_ids), 10)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return RetrievalMetrics(recall(5), recall(10), reciprocal_rank, dcg / idcg if idcg else 0.0)


def aggregate_metrics(values: list[RetrievalMetrics]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty evaluation set")
    return {
        "recall_at_5": sum(item.recall_at_5 for item in values) / len(values),
        "recall_at_10": sum(item.recall_at_10 for item in values) / len(values),
        "mrr": sum(item.mrr for item in values) / len(values),
        "ndcg_at_10": sum(item.ndcg_at_10 for item in values) / len(values),
    }


async def benchmark_retrieval_strategies(
    dataset: GoldenDataset,
    retriever: Any,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare BM25, dense, RRF hybrid and RRF plus reranking on identical cases."""
    strategies = ("bm25", "dense", "hybrid", "hybrid_rerank")
    report: dict[str, Any] = {
        "dataset": dataset.dataset_name,
        "query_count": len(dataset.questions),
        "top_k": top_k,
        "strategies": {},
    }
    for strategy in strategies:
        per_question = []
        elapsed = 0.0
        for case in dataset.questions:
            started = time.perf_counter()
            hits = await retriever.retrieve(case.question, top_k, strategy)  # type: ignore[arg-type]
            elapsed += time.perf_counter() - started
            metrics = retrieval_metrics(
                set(case.relevant_chunk_ids), [hit.chunk_id for hit in hits]
            )
            per_question.append(
                {
                    "question_id": case.question_id,
                    **metrics.__dict__,
                    "retrieved_chunk_ids": [hit.chunk_id for hit in hits],
                }
            )
        report["strategies"][strategy] = {
            **aggregate_metrics(
                [
                    RetrievalMetrics(**{key: row[key] for key in RetrievalMetrics.__annotations__})
                    for row in per_question
                ]
            ),
            "mean_latency_ms": elapsed * 1000 / len(dataset.questions),
            "per_question": per_question,
        }
    report["target_recall_at_10"] = 0.70
    report["hybrid_recall_at_10_target_met"] = report["strategies"]["hybrid"]["recall_at_10"] > 0.70
    return report


def benchmark_embedding_candidates(
    model_names: list[str],
    questions: list[GoldenQuestion],
    corpus: list[RetrievalDocument],
    *,
    device: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Evaluate candidate bi-encoders, reporting dimension, encode latency and peak GPU memory."""
    if not model_names or not corpus:
        raise ValueError("At least one embedding model and one corpus chunk are required")
    results = []
    all_texts = [document.text for document in corpus]
    query_texts = [question.question for question in questions]
    for model_name in model_names:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Install the Phase 4 requirements to benchmark embedding models"
            ) from error
        encoder = SentenceTransformerEmbedder(model_name, device=device, revision=revision)
        load_started = time.perf_counter()
        encoder._load()
        load_seconds = time.perf_counter() - load_started
        gpu_available = bool(torch.cuda.is_available() and (device is None or "cuda" in device))
        if gpu_available:
            torch.cuda.reset_peak_memory_stats()
        encode_started = time.perf_counter()
        corpus_vectors = encoder.encode_many(all_texts)
        corpus_encode_seconds = time.perf_counter() - encode_started
        query_vectors = []
        query_elapsed = 0.0
        for query in query_texts:
            query_started = time.perf_counter()
            query_vectors.append(encoder.encode_many([query])[0])
            query_elapsed += time.perf_counter() - query_started
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if gpu_available else None
        per_query = []
        for question, query_vector in zip(questions, query_vectors, strict=True):
            ranked = sorted(
                range(len(corpus)),
                key=lambda index: (
                    -_cosine(query_vector, corpus_vectors[index]),
                    corpus[index].chunk_id,
                ),
            )
            ids = [corpus[index].chunk_id for index in ranked]
            per_query.append(retrieval_metrics(set(question.relevant_chunk_ids), ids))
        results.append(
            {
                "model": model_name,
                "revision": revision,
                "dimensions": len(corpus_vectors[0]),
                "device": device or ("cuda" if gpu_available else "cpu"),
                "model_load_seconds": load_seconds,
                "corpus_encode_seconds": corpus_encode_seconds,
                "mean_query_encode_ms": query_elapsed * 1000 / max(1, len(query_texts)),
                "peak_gpu_memory_mb": peak_gpu_mb,
                "metrics": aggregate_metrics(per_query),
            }
        )
    return {"models": results, "selected_model": None, "selection_requires_review": True}


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def benchmark_reranker_candidates(
    model_names: list[str],
    questions: list[GoldenQuestion],
    candidate_hits: dict[str, list[RetrievalHit]],
    *,
    device: str | None = None,
    top_k: int = 10,
    revision: str | None = None,
) -> dict[str, Any]:
    """Measure the reranking gain/cost on the same hybrid candidates."""
    if set(candidate_hits) != {question.question_id for question in questions}:
        raise ValueError(
            "Provide exactly one shared hybrid candidate list for every Golden question"
        )
    results = []
    for model_name in model_names:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Install the Phase 4 requirements to benchmark rerankers") from error
        reranker = SentenceTransformerReranker(model_name, device=device, revision=revision)
        reranker._load()
        gpu_available = bool(torch.cuda.is_available() and (device is None or "cuda" in device))
        if gpu_available:
            torch.cuda.reset_peak_memory_stats()
        baseline, reranked = [], []
        elapsed = 0.0
        for question in questions:
            hits = candidate_hits[question.question_id]
            baseline.append(
                retrieval_metrics(
                    set(question.relevant_chunk_ids), [hit.chunk_id for hit in hits[:top_k]]
                )
            )
            eligible = [hit for hit in hits if hit.section_type not in {"table", "formula"}]
            structural = [hit for hit in hits if hit.section_type in {"table", "formula"}]
            started = time.perf_counter()
            scores = reranker.score(question.question, [hit.text for hit in eligible])
            elapsed += time.perf_counter() - started
            ordered = sorted(
                zip(eligible, scores, strict=True), key=lambda pair: (-pair[1], pair[0].chunk_id)
            )
            ranked_ids = [hit.chunk_id for hit, _ in ordered] + [hit.chunk_id for hit in structural]
            reranked.append(retrieval_metrics(set(question.relevant_chunk_ids), ranked_ids[:top_k]))
        results.append(
            {
                "model": model_name,
                "device": device or ("cuda" if gpu_available else "cpu"),
                "mean_latency_ms": elapsed * 1000 / len(questions),
                "peak_gpu_memory_mb": (
                    torch.cuda.max_memory_allocated() / (1024 * 1024) if gpu_available else None
                ),
                "baseline_hybrid": aggregate_metrics(baseline),
                "hybrid_reranked": aggregate_metrics(reranked),
                "recall_at_10_gain": (
                    aggregate_metrics(reranked)["recall_at_10"]
                    - aggregate_metrics(baseline)["recall_at_10"]
                ),
            }
        )
    return {"models": results, "selected_model": None, "selection_requires_review": True}


def evaluate_ragas(rows_by_strategy: dict[str, list[dict[str, Any]]], *, llm: Any, embeddings: Any):
    """Run the four pinned RAGAS 0.2.x metrics on generated answers and references."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to run the RAGAS evaluation") from error
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    results = {}
    for strategy, rows in rows_by_strategy.items():
        required = {"question", "answer", "contexts", "ground_truth"}
        if not rows or any(not required.issubset(row) for row in rows):
            raise ValueError(f"RAGAS rows for {strategy} must contain {sorted(required)}")
        result = evaluate(Dataset.from_list(rows), metrics=metrics, llm=llm, embeddings=embeddings)
        results[strategy] = dict(result)
    return results


async def benchmark_retrieval_router(
    dataset: GoldenDataset, retriever: Any, *, top_k: int = 10
) -> dict[str, Any]:
    """Compare hybrid retrieval with and without the deterministic complexity router."""
    baseline_scores, routed_scores = [], []
    baseline_time = routed_time = 0.0
    for question in dataset.questions:
        started = time.perf_counter()
        baseline = await retriever.retrieve(question.question, top_k, "hybrid", route=False)
        baseline_time += time.perf_counter() - started
        started = time.perf_counter()
        routed = await retriever.retrieve(question.question, top_k, "hybrid", route=True)
        routed_time += time.perf_counter() - started
        relevant = set(question.relevant_chunk_ids)
        baseline_scores.append(retrieval_metrics(relevant, [hit.chunk_id for hit in baseline]))
        routed_scores.append(retrieval_metrics(relevant, [hit.chunk_id for hit in routed]))
    count = len(dataset.questions)
    return {
        "baseline_hybrid": aggregate_metrics(baseline_scores),
        "routed_hybrid": aggregate_metrics(routed_scores),
        "baseline_mean_latency_ms": baseline_time * 1000 / count,
        "routed_mean_latency_ms": routed_time * 1000 / count,
    }


def golden_case_to_query(case: GoldenQuestion) -> str:
    """Return the human-authored retrieval query without normalization."""
    return case.question
