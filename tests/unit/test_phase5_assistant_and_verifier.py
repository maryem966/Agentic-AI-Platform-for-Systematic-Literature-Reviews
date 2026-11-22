"""Unit and pipeline tests for Phase 5: Research Assistant Agent and 4-Level Claim Verifier."""

from __future__ import annotations

import pytest

from src.agents.research_assistant_agent import (
    FieldExtractionItem,
    PublicationExtractionLLMOutput,
    ResearchAssistantAgent,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.claim import Claim, ClaimEvidenceLink
from src.domain.entities.evidence import Evidence
from src.domain.entities.extraction_record import ExtractionRecord
from src.domain.entities.publication import Publication
from src.domain.entities.research_protocol import ResearchProtocol, ResearchQuestion
from src.evidence.verifier import (
    FourLevelClaimVerifier,
    NLIEvaluationOutput,
)


def _make_chunk(
    *,
    id: str,
    publication_id: str,
    text: str,
    document_id: str = "doc-1",
    page: int = 1,
    section: str = "Results",
    section_type: str = "text",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        id=id,
        publication_id=publication_id,
        document_id=document_id,
        text=text,
        char_start=0,
        char_end=len(text),
        page=page,
        section=section,
        section_type=section_type,  # type: ignore[arg-type]
        chunk_index=chunk_index,
        embedding_model="all-minilm",
        embedding_version="1",
    )


class MockEvidencePort:
    def __init__(self, existing_evidences: list[Evidence] | None = None) -> None:
        self.evidences = {ev.id: ev for ev in (existing_evidences or [])}

    async def find_evidence_by_id(self, evidence_id: str) -> Evidence | None:
        return self.evidences.get(evidence_id)


# =============================================================================
# Level 1 — Existence Tests
# =============================================================================

@pytest.mark.asyncio
async def test_claim_without_evidence_fails_level1_existence() -> None:
    verifier = FourLevelClaimVerifier()
    claim = Claim(
        text="Les transformers surpassent les RNNs sur les séries temporelles.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Results",
    )
    verification = await verifier.verify(claim, [])
    assert verification.status == "unsupported"
    assert verification.verification_method == "existence"
    assert "Invariant violé" in (verification.reason or "")
    assert verification.requires_human_review is True
    assert verification.is_acceptable is False


@pytest.mark.asyncio
async def test_missing_evidence_id_fails_level1_existence() -> None:
    port = MockEvidencePort([])
    verifier = FourLevelClaimVerifier(evidence_port=port)
    claim = Claim(
        text="L'attention multi-tête permet une parallélisation efficace.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Results",
    )
    ev = Evidence(
        id="missing-ev-id",
        publication_id="pub-1",
        chunk_id="chunk-1",
        text="Multi-head attention enables parallelization.",
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.9,
        research_id="res-1",
    )
    verification = await verifier.verify(claim, [ev])
    assert verification.status == "unsupported"
    assert verification.verification_method == "existence"
    assert "missing-ev-id" in verification.missing_evidence_ids
    assert verification.requires_human_review is True


# =============================================================================
# Level 2 — Provenance Tests
# =============================================================================

@pytest.mark.asyncio
async def test_chunk_publication_mismatch_fails_level2_provenance() -> None:
    source_chunk = _make_chunk(
        id="chunk-1",
        publication_id="pub-DIFFERENT",
        text="Multi-head attention enables parallelization.",
    )
    ev = Evidence(
        id="ev-1",
        publication_id="pub-1",
        chunk_id="chunk-1",
        text="Multi-head attention enables parallelization.",
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.9,
        research_id="res-1",
    )
    port = MockEvidencePort([ev])
    verifier = FourLevelClaimVerifier(
        evidence_port=port,
        chunk_resolver=lambda cid: source_chunk,
    )
    claim = Claim(
        text="L'attention multi-tête permet une parallélisation efficace.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Results",
    )
    verification = await verifier.verify(claim, [ev])
    assert verification.status == "unsupported"
    assert verification.verification_method == "provenance"
    assert "Incohérence de publication" in (verification.reason or "")
    assert verification.requires_human_review is True


@pytest.mark.asyncio
async def test_non_verbatim_evidence_fails_level2_provenance() -> None:
    source_chunk = _make_chunk(
        id="chunk-1",
        publication_id="pub-1",
        text="The empirical study demonstrated a 15% reduction in latency.",
    )
    # Paraphrased / modified evidence text not in chunk
    ev = Evidence(
        id="ev-1",
        publication_id="pub-1",
        chunk_id="chunk-1",
        text="The experiment showed 25% lower delay.",  # Not in source chunk
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.9,
        research_id="res-1",
    )
    port = MockEvidencePort([ev])
    verifier = FourLevelClaimVerifier(
        evidence_port=port,
        chunk_resolver=lambda cid: source_chunk,
    )
    claim = Claim(
        text="La méthode réduit considérablement la latence d'exécution.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Results",
    )
    verification = await verifier.verify(claim, [ev])
    assert verification.status == "unsupported"
    assert verification.verification_method == "provenance"
    assert "n'est pas un passage littéral exact" in (verification.reason or "")


# =============================================================================
# Level 3 — Semantic Similarity Tests
# =============================================================================

@pytest.mark.asyncio
async def test_unrelated_passage_fails_level3_semantic_similarity() -> None:
    chunk_text = "Photosynthesis in C4 plants operates via the Hatch-Slack pathway in bundle sheath cells."
    source_chunk = _make_chunk(
        id="chunk-1",
        publication_id="pub-1",
        text=chunk_text,
    )
    ev = Evidence(
        id="ev-1",
        publication_id="pub-1",
        chunk_id="chunk-1",
        text=chunk_text,
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.95,
        research_id="res-1",
    )
    port = MockEvidencePort([ev])

    # Custom embedder returning orthogonal vectors
    def orthogonal_embedder(text: str) -> list[float]:
        if "Photosynthesis" in text:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]

    verifier = FourLevelClaimVerifier(
        evidence_port=port,
        chunk_resolver=lambda cid: source_chunk,
        embedder=orthogonal_embedder,
        similarity_threshold=0.60,
    )
    claim = Claim(
        text="Quantum key distribution provides unconditional cryptographic secrecy.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Results",
    )
    verification = await verifier.verify(claim, [ev])
    assert verification.status == "unsupported"
    assert verification.verification_method == "semantic"
    assert verification.semantic_similarity_score == 0.0
    assert "Similarité sémantique insuffisante" in (verification.reason or "")
    assert verification.requires_human_review is True


# =============================================================================
# Level 4 — Entailment and Contradiction Tests
# =============================================================================

@pytest.mark.asyncio
async def test_entailment_accepts_claim_without_human_review() -> None:
    chunk_text = "The proposed hybrid retrieval pipeline achieves 0.88 Recall@10 on benchmark data."
    source_chunk = _make_chunk(
        id="chunk-1",
        publication_id="pub-1",
        text=chunk_text,
    )
    ev = Evidence(
        id="ev-1",
        publication_id="pub-1",
        chunk_id="chunk-1",
        text=chunk_text,
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.95,
        research_id="res-1",
    )
    port = MockEvidencePort([ev])

    def matched_embedder(text: str) -> list[float]:
        return [0.7071, 0.7071, 0.0, 0.0]

    def mock_nli(premise: str, hypothesis: str) -> NLIEvaluationOutput:
        return NLIEvaluationOutput(
            label="entailment",
            confidence=0.95,
            explanation="The empirical result confirms the claimed recall performance.",
        )

    verifier = FourLevelClaimVerifier(
        evidence_port=port,
        chunk_resolver=lambda cid: source_chunk,
        embedder=matched_embedder,
        similarity_threshold=0.50,
        nli_evaluator=mock_nli,
    )
    claim = Claim(
        text="The hybrid retrieval pipeline attains superior Recall@10 exceeding 0.85.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Results",
    )
    verification = await verifier.verify(claim, [ev])
    assert verification.status == "supported"
    assert verification.verification_method == "entailment"
    assert verification.entailment_label == "entailment"
    assert verification.is_acceptable is True
    assert verification.requires_human_review is False


@pytest.mark.asyncio
async def test_contradiction_detects_conflict_and_requires_review() -> None:
    chunk_text = "The proposed approach fails completely and yields no improvement over baseline."
    source_chunk = _make_chunk(
        id="chunk-1",
        publication_id="pub-1",
        text=chunk_text,
    )
    ev = Evidence(
        id="ev-1",
        publication_id="pub-1",
        chunk_id="chunk-1",
        text=chunk_text,
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.95,
        research_id="res-1",
    )
    port = MockEvidencePort([ev])

    def matched_embedder(text: str) -> list[float]:
        return [0.7071, 0.7071, 0.0, 0.0]

    def mock_contradiction_nli(premise: str, hypothesis: str) -> NLIEvaluationOutput:
        return NLIEvaluationOutput(
            label="contradiction",
            confidence=0.98,
            explanation="The source text refutes the claimed improvement.",
        )

    verifier = FourLevelClaimVerifier(
        evidence_port=port,
        chunk_resolver=lambda cid: source_chunk,
        embedder=matched_embedder,
        similarity_threshold=0.50,
        nli_evaluator=mock_contradiction_nli,
    )
    claim = Claim(
        text="The proposed approach significantly improves performance over baseline.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Results",
    )
    verification = await verifier.verify(claim, [ev])
    assert verification.status == "contradicted"
    assert verification.verification_method == "entailment"
    assert verification.entailment_label == "contradiction"
    assert "ev-1" in verification.contradicting_evidence_ids
    assert verification.is_acceptable is False
    assert verification.requires_human_review is True


@pytest.mark.asyncio
async def test_find_contradicting_evidence_across_registry() -> None:
    claim = Claim(
        text="Deep learning consistently outperforms traditional statistical methods on all tasks.",
        rq_id="RQ1",
        report_version_id="rep-1",
        section="Discussion",
    )
    supporting_ev = Evidence(
        id="ev-supp",
        publication_id="pub-1",
        chunk_id="c-1",
        text="Deep neural architectures achieved state of the art results across test sets.",
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.9,
        research_id="res-1",
    )
    contradicting_ev = Evidence(
        id="ev-contra",
        publication_id="pub-2",
        chunk_id="c-2",
        text="Contrary to expectations, traditional linear models yielded higher accuracy than deep models.",
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.92,
        research_id="res-1",
    )

    def mock_nli(premise: str, hypothesis: str) -> NLIEvaluationOutput:
        if "Contrary to expectations" in premise:
            return NLIEvaluationOutput(label="contradiction", explanation="Refutes superiority on all tasks.")
        return NLIEvaluationOutput(label="entailment", explanation="Supports claim.")

    verifier = FourLevelClaimVerifier(nli_evaluator=mock_nli)
    contradictions = await verifier.find_contradicting_evidence(
        claim, [supporting_ev, contradicting_ev]
    )
    assert len(contradictions) == 1
    assert contradictions[0].id == "ev-contra"


# =============================================================================
# Research Assistant Agent Tests
# =============================================================================

@pytest.mark.asyncio
async def test_assistant_extracts_structured_fields_and_literal_evidence() -> None:
    pub = Publication(
        id="pub-100",
        title="Scalable Graph Neural Networks for Recommendation Systems",
        abstract="We propose a scalable GNN approach.",
        source="arxiv",
    )
    chunk1 = _make_chunk(
        id="chunk-101",
        publication_id=pub.id,
        text="The proposed method uses neighborhood sampling and multi-head attention. We evaluated our model on the MovieLens dataset using F1-score metric.",
        section="Methods",
        section_type="text",
        chunk_index=0,
    )
    chunk2 = _make_chunk(
        id="chunk-102",
        publication_id=pub.id,
        text="The experimental result outperforms all previous baselines by 12%. However, the main limitation remains GPU memory consumption on massive graphs.",
        section="Results",
        section_type="text",
        chunk_index=1,
    )

    agent = ResearchAssistantAgent()
    record, evidences = await agent.extract_from_publication(
        publication=pub,
        chunks=[chunk1, chunk2],
        research_id="res-test",
        agent_execution_id="exec-1",
    )

    assert isinstance(record, ExtractionRecord)
    assert record.publication_id == pub.id
    assert record.is_anchored is True
    assert len(record.source_chunk_ids) > 0

    # Ensure all extracted evidences are literal exact passages from their respective chunks
    chunk_map = {chunk1.id: chunk1.text, chunk2.id: chunk2.text}
    assert len(evidences) >= 2
    for ev in evidences:
        assert ev.chunk_id in chunk_map
        assert ev.text in chunk_map[ev.chunk_id]
        assert ev.publication_id == pub.id
        assert ev.extracted_by_agent == "research_assistant"


@pytest.mark.asyncio
async def test_assistant_formulates_claims_with_mandatory_evidence_links() -> None:
    protocol = ResearchProtocol(
        research_id="res-1",
        research_questions=[
            ResearchQuestion(text="RQ1: What architectures achieve the highest recommendation accuracy?"),
            ResearchQuestion(text="RQ2: What computational limitations are identified in recent literature?"),
        ],
        inclusion_criteria=["Criterion 1", "Criterion 2"],
        exclusion_criteria=["Exclusion 1", "Exclusion 2"],
    )
    ev1 = Evidence(
        id="ev-rec-1",
        publication_id="pub-1",
        chunk_id="chunk-1",
        text="Graph Convolutional Networks achieve 0.93 accuracy on recommendation benchmarks.",
        evidence_type="result",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.9,
        research_id="res-1",
    )
    ev2 = Evidence(
        id="ev-rec-2",
        publication_id="pub-2",
        chunk_id="chunk-2",
        text="Memory consumption during full graph training exceeds 32GB VRAM.",
        evidence_type="limitation",
        extracted_by_agent="research_assistant",
        extraction_confidence=0.88,
        research_id="res-1",
    )

    agent = ResearchAssistantAgent()
    claims_with_links = await agent.formulate_claims(
        research_id="res-1",
        report_version_id="rep-v1",
        research_protocol=protocol,
        evidences=[ev1, ev2],
    )

    assert len(claims_with_links) >= 2
    for claim, links in claims_with_links:
        assert isinstance(claim, Claim)
        assert len(claim.text) >= 10
        assert len(links) >= 1
        for link in links:
            assert isinstance(link, ClaimEvidenceLink)
            assert link.claim_id == claim.id
            assert link.evidence_id in {"ev-rec-1", "ev-rec-2"}


@pytest.mark.asyncio
async def test_assistant_with_structured_llm_output() -> None:
    class MockStructuredLLM:
        async def invoke_structured(self, messages, response_model, **kwargs):
            if response_model is PublicationExtractionLLMOutput:
                return PublicationExtractionLLMOutput(
                    fields=[
                        FieldExtractionItem(
                            field_name="method",
                            value="Hybrid RRF Fusion",
                            verbatim_quote="Hybrid reciprocal rank fusion combines sparse and dense indices.",
                            chunk_id="chunk-1",
                            evidence_type="method",
                            page=2,
                            section="Methods",
                            confidence=0.95,
                        ),
                        FieldExtractionItem(
                            field_name="result",
                            value="Recall@10 of 0.89",
                            verbatim_quote="The resulting pipeline achieved a measured Recall@10 of 0.89.",
                            chunk_id="chunk-1",
                            evidence_type="result",
                            page=4,
                            section="Results",
                            confidence=0.92,
                        ),
                    ],
                    overall_confidence=0.94,
                )
            raise ValueError(f"Unexpected response model: {response_model}")

    pub = Publication(id="pub-1", title="Hybrid Retrieval in SLRs", source="test")
    full_text = (
        "Hybrid reciprocal rank fusion combines sparse and dense indices. "
        "The resulting pipeline achieved a measured Recall@10 of 0.89."
    )
    chunk = _make_chunk(
        id="chunk-1",
        publication_id="pub-1",
        text=full_text,
    )

    agent = ResearchAssistantAgent(llm=MockStructuredLLM())
    record, evidences = await agent.extract_from_publication(
        publication=pub,
        chunks=[chunk],
        research_id="res-1",
        agent_execution_id="exec-1",
    )

    assert record.extracted_fields["method"] == "Hybrid RRF Fusion"
    assert record.extracted_fields["result"] == "Recall@10 of 0.89"
    assert len(evidences) == 2
    assert evidences[0].text == "Hybrid reciprocal rank fusion combines sparse and dense indices."
    assert evidences[0].evidence_type == "method"
    assert evidences[1].text == "The resulting pipeline achieved a measured Recall@10 of 0.89."
    assert evidences[1].evidence_type == "result"
