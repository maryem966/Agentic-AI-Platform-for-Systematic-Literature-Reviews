"""Run Phase 4 benchmarks from an annotated Golden Dataset and indexed chunks."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import settings
from src.infrastructure.llm.ollama_adapter import OllamaAdapter
from src.infrastructure.persistence.database import create_session_factory
from src.infrastructure.persistence.repositories import ChunkRepo
from src.orchestration.postgres_checkpointer import configure_postgres_event_loop
from src.rag.evaluation import (
    benchmark_embedding_candidates,
    benchmark_reranker_candidates,
    benchmark_retrieval_router,
    benchmark_retrieval_strategies,
    evaluate_ragas,
)
from src.rag.golden_dataset import GoldenDataset
from src.rag.retrieval import (
    BM25Retriever,
    HybridRetriever,
    InMemoryDenseRetriever,
    RetrievalDocument,
    SentenceTransformerEmbedder,
    SentenceTransformerReranker,
)


def _collection_name(model: str, revision: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.casefold()).strip("_")
    return f"chunks_{slug[:32]}_{revision[:8]}"


def _model_spec(value: str) -> tuple[str, str]:
    if "@" not in value:
        raise ValueError(
            f"Model candidate {value!r} must include an immutable model revision as MODEL@COMMIT"
        )
    model, revision = value.rsplit("@", 1)
    if not model.strip() or not revision.strip():
        raise ValueError("Model candidate must contain both a model ID and an immutable revision")
    return model, revision


async def run(args: argparse.Namespace) -> dict[str, Any]:
    embedding_candidates = [_model_spec(value) for value in args.embedding_model]
    reranker_candidates = [_model_spec(value) for value in args.reranker_model]
    dataset_path = args.dataset.resolve()
    dataset = GoldenDataset.load(dataset_path)
    dataset.validate_pdf_assets(dataset_path.parent)
    publication_ids = sorted(
        {
            publication.publication_id
            for question in dataset.questions
            for publication in question.publications
        }
    )
    session_factory = create_session_factory()
    try:
        async with session_factory() as session:
            chunks = await ChunkRepo(session).list_for_publications(publication_ids)
    finally:
        engine = session_factory.kw.get("bind")
        if engine is not None:
            await engine.dispose()
    if not chunks:
        raise ValueError("No indexed chunks found for the Golden Dataset publications")
    chunk_ids = {chunk.id for chunk in chunks}
    expected_ids = {
        chunk_id for question in dataset.questions for chunk_id in question.relevant_chunk_ids
    }
    missing = sorted(expected_ids - chunk_ids)
    if missing:
        raise ValueError(
            f"Annotated relevant chunk IDs are missing from PostgreSQL: {missing[:10]}"
        )
    documents = [
        RetrievalDocument(
            chunk_id=chunk.id,
            text=chunk.text,
            section_type=chunk.section_type,
            metadata={"publication_id": chunk.publication_id, "page": chunk.page},
        )
        for chunk in chunks
    ]
    report: dict[str, Any] = {
        "dataset_name": dataset.dataset_name,
        "question_count": len(dataset.questions),
        "corpus_chunk_count": len(documents),
        "embedding_candidates": [],
        "benchmark_by_embedding_model": [],
        "ragas_by_strategy": {},
        "selected_models": {"embedding": None, "reranker": None},
        "status": "completed_without_model_selection",
    }
    for model, revision in embedding_candidates:
        report["embedding_candidates"].extend(
            benchmark_embedding_candidates(
                [model], dataset.questions, documents, device=args.device, revision=revision
            )["models"]
        )
    bm25 = BM25Retriever(documents)
    llm = OllamaAdapter()

    for embed_model, embedding_revision in embedding_candidates:
        embedder = SentenceTransformerEmbedder(
            embed_model, device=args.device, revision=embedding_revision
        )
        embedder._load()
        vectors = embedder.encode_many([document.text for document in documents])
        dense = InMemoryDenseRetriever(documents, vectors, embedder)
        per_reranker = []
        for reranker_model, reranker_revision in reranker_candidates:
            reranker = SentenceTransformerReranker(
                reranker_model,
                device=args.device,
                batch_size=args.batch_size,
                revision=reranker_revision,
            )
            retriever = HybridRetriever(bm25, dense, reranker, rrf_k=args.rrf_k)
            strategy_report = await benchmark_retrieval_strategies(
                dataset, retriever, top_k=args.top_k
            )
            strategy_report["router"] = await benchmark_retrieval_router(
                dataset, retriever, top_k=args.top_k
            )
            rerank_candidates = {}
            for question in dataset.questions:
                candidates = await retriever.retrieve(question.question, args.candidate_k, "hybrid")
                rerank_candidates[question.question_id] = candidates
            rerank_report = benchmark_reranker_candidates(
                [reranker_model],
                dataset.questions,
                rerank_candidates,
                device=args.device,
                top_k=args.top_k,
                revision=reranker_revision,
            )
            per_reranker.append(
                {
                    "model": reranker_model,
                    "revision": reranker_revision,
                    **rerank_report["models"][0],
                }
            )
            if not args.skip_ragas:
                ragas_rows = await generate_ragas_rows(dataset, retriever, llm, args.top_k)
                ragas_report = await asyncio.to_thread(
                    evaluate_ragas,
                    ragas_rows,
                    llm=_ragas_llm(),
                    embeddings=_ragas_embeddings(),
                )
                report["ragas_by_strategy"][f"{embed_model}::{reranker_model}"] = ragas_report
        report["benchmark_by_embedding_model"].append(
            {
                "embedding_model": embed_model,
                "embedding_revision": embedding_revision,
                "retrieval": strategy_report,
                "reranker_benchmark": per_reranker,
            }
        )
    report["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.select_embedding or args.select_reranker:
        selected_embedding = _model_spec(args.select_embedding) if args.select_embedding else None
        selected_reranker = _model_spec(args.select_reranker) if args.select_reranker else None
        if selected_embedding and selected_embedding not in embedding_candidates:
            raise ValueError("Selected embedding MODEL@REV must be one of the tested candidates")
        if selected_reranker and selected_reranker not in reranker_candidates:
            raise ValueError("Selected reranker MODEL@REV must be one of the tested candidates")
        report["selected_models"] = {
            "embedding": {
                "model": selected_embedding[0],
                "revision": selected_embedding[1],
            }
            if selected_embedding
            else None,
            "reranker": {
                "model": selected_reranker[0],
                "revision": selected_reranker[1],
            }
            if selected_reranker
            else None,
        }
        if selected_embedding:
            selected_measurement = next(
                item
                for item in report["embedding_candidates"]
                if item["model"] == selected_embedding[0]
                and item["revision"] == selected_embedding[1]
            )
            report["selected_models"]["embedding"]["dimensions"] = selected_measurement[
                "dimensions"
            ]
            report["selected_models"]["embedding"]["qdrant_collection"] = _collection_name(
                selected_embedding[0], selected_embedding[1]
            )
        report["status"] = "completed_with_human_model_selection"
    return report


async def generate_ragas_rows(
    dataset: GoldenDataset, retriever: HybridRetriever, llm: OllamaAdapter, top_k: int
) -> dict[str, list[dict[str, Any]]]:
    rows = {strategy: [] for strategy in ("bm25", "dense", "hybrid", "hybrid_rerank")}
    for question in dataset.questions:
        for strategy in rows:
            hits = await retriever.retrieve(question.question, top_k, strategy)
            contexts = [hit.text for hit in hits]
            answer = await llm.invoke(
                [
                    {
                        "role": "system",
                        "content": "Answer using only the supplied untrusted source passages. Cite their chunk IDs. If evidence is insufficient, say so.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": question.question,
                                "passages_untrusted": [
                                    {"chunk_id": hit.chunk_id, "text": hit.text} for hit in hits
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0,
            )
            rows[strategy].append(
                {
                    "question": question.question,
                    "answer": answer.content,
                    "contexts": contexts,
                    "ground_truth": question.reference_answer,
                }
            )
    return rows


def _ragas_llm() -> Any:
    try:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
    except ImportError as error:
        raise RuntimeError("Install the Phase 4 RAGAS/LangChain requirements") from error
    return LangchainLLMWrapper(
        ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_default_model)
    )


def _ragas_embeddings() -> Any:
    try:
        from langchain_ollama import OllamaEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as error:
        raise RuntimeError("Install the Phase 4 RAGAS/LangChain requirements") from error
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(base_url=settings.ollama_base_url, model=settings.embedding_model)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/phase4_benchmark.json"))
    parser.add_argument(
        "--embedding-model",
        action="append",
        required=True,
        help="Candidate Sentence Transformers model; repeat for multiple candidates",
    )
    parser.add_argument(
        "--reranker-model",
        action="append",
        required=True,
        help="Candidate cross-encoder; repeat for multiple candidates",
    )
    parser.add_argument(
        "--select-embedding", default=None, help="Selected tested candidate MODEL@COMMIT"
    )
    parser.add_argument(
        "--select-reranker", default=None, help="Selected tested candidate MODEL@COMMIT"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--skip-ragas", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_postgres_event_loop()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["status"] == "completed_with_human_model_selection":
        selected = report["selected_models"]
        lines = []
        if selected["embedding"]:
            lines.extend(
                [
                    "EMBEDDING_PROVIDER=sentence-transformers",
                    f"EMBEDDING_MODEL={selected['embedding']['model']}",
                    f"EMBEDDING_MODEL_REVISION={selected['embedding']['revision']}",
                    f"EMBEDDING_VECTOR_SIZE={selected['embedding']['dimensions']}",
                    f"QDRANT_CHUNKS_COLLECTION={selected['embedding']['qdrant_collection']}",
                ]
            )
        if selected["reranker"]:
            lines.extend(
                [
                    f"RERANKER_MODEL={selected['reranker']['model']}",
                    f"RERANKER_MODEL_REVISION={selected['reranker']['revision']}",
                ]
            )
        args.output.with_suffix(".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Benchmark report saved to {args.output}")
    for model_result in report["benchmark_by_embedding_model"]:
        hybrid = model_result["retrieval"]["strategies"]["hybrid"]
        print(
            f"{model_result['embedding_model']}: Hybrid Recall@10="
            f"{hybrid['recall_at_10']:.3f}, MRR={hybrid['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
