from __future__ import annotations

import pytest

from src.agents.screening_agent import ScreeningAgent, ScreeningOutput
from src.domain.entities.publication import Publication
from src.domain.entities.research_protocol import ResearchProtocol
from src.rag.chunking import DocumentChunker, PageText
from src.rag.document_ingestion import PDFIngestion


def test_chunking_uses_overlap_and_document_offsets() -> None:
    page = PageText(1, " ".join(f"word{i}" for i in range(700)))
    chunks = DocumentChunker(size=200, overlap=20, minimum=50).split(
        [page],
        document_id="doc",
        publication_id="pub",
        embedding_model="test",
        embedding_version="1",
    )
    assert len(chunks) > 1
    assert chunks[0].char_start == 0
    assert chunks[1].char_start < chunks[0].char_end
    assert chunks[-1].char_end == len(page.text)


def test_short_chunks_are_merged() -> None:
    chunker = DocumentChunker(size=20, overlap=2, minimum=10)
    chunks = chunker.split(
        [PageText(1, "alpha beta gamma")],
        document_id="d",
        publication_id="p",
        embedding_model="test",
        embedding_version="1",
    )
    assert len(chunks) == 1
    assert "alpha beta gamma" in chunks[0].text


def test_missing_abstract_is_uncertain_and_never_auto_excluded() -> None:
    publication = Publication(title="A study", abstract=None, source="test")
    protocol = ResearchProtocol(
        research_id="r",
        inclusion_criteria=["include AI"],
        exclusion_criteria=["Exclude wrong population"],
    )
    decision = ScreeningAgent.deterministic_decision(publication, protocol)
    assert decision is not None
    assert decision.decision == "uncertain"
    assert decision.requires_human_arbitration


def test_ambiguous_abstract_uses_structured_llm_output() -> None:
    class FakeLLM:
        async def invoke_structured(self, messages, response_model, **kwargs):
            assert response_model is ScreeningOutput
            assert messages[0]["role"] == "system"
            assert "untrusted" in messages[1]["content"]
            return ScreeningOutput(
                decision="include",
                justification="Abstract matches the stated inclusion criteria.",
                criteria_applied=["include AI"],
                confidence=0.82,
            )

    publication = Publication(
        title="AI in research", abstract="This paper examines AI in education.", source="test"
    )
    protocol = ResearchProtocol(
        research_id="r",
        inclusion_criteria=["include AI"],
        exclusion_criteria=["Exclude wrong population"],
    )
    pending = ScreeningAgent(FakeLLM()).screen(publication, protocol)
    with pytest.raises(StopIteration) as completed:
        pending.send(None)
    decision = completed.value.value
    assert decision.decision == "include"
    assert decision.agent == "llm"


def test_pdf_signature_is_checked_before_parsing() -> None:
    with pytest.raises(ValueError, match="signature"):
        PDFIngestion().extract(b"not a pdf")
