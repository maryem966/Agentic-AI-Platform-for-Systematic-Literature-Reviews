"""
Tests unitaires : Entités du domaine SLR-Nexus.
Aucune dépendance externe — pas de LLM, pas de base de données.
"""
from __future__ import annotations

import pytest
from datetime import datetime

from src.domain.entities.research_project import ResearchProject
from src.domain.entities.research_protocol import (
    DateRange,
    ProtocolModification,
    ResearchProtocol,
    ResearchQuestion,
)
from src.domain.entities.search_run import SearchQuery, SearchRun
from src.domain.entities.publication import Author, Publication
from src.domain.entities.full_text_document import FullTextDocument
from src.domain.entities.screening_decision import ScreeningDecision
from src.domain.entities.chunk import Chunk
from src.domain.entities.extraction_record import ExtractionRecord
from src.domain.entities.evidence import Evidence
from src.domain.entities.claim import Claim, ClaimEvidenceLink
from src.domain.entities.report_version import ReportVersion
from src.domain.entities.agent_execution import AgentExecution
from src.domain.entities.human_review import HumanReview


# ============================================================
# ResearchProject
# ============================================================

class TestResearchProject:

    def test_create_project_default_status(self):
        project = ResearchProject(
            title="SLR on Federated Learning",
            topic="Survey of federated learning methods for healthcare applications",
            created_by="user_001",
        )
        assert project.status == "draft"
        assert project.protocol_id is None
        assert project.id is not None

    def test_valid_transition(self):
        project = ResearchProject(
            title="Test SLR",
            topic="Testing systematic review transitions in the system",
            created_by="user_001",
        )
        new_project = project.transition_to("protocol_defined")
        assert new_project.status == "protocol_defined"
        assert project.status == "draft"  # Immuabilité logique

    def test_invalid_transition_raises(self):
        project = ResearchProject(
            title="Test SLR",
            topic="Testing systematic review transitions in the system",
            created_by="user_001",
        )
        with pytest.raises(ValueError, match="Transition invalide"):
            project.transition_to("completed")

    def test_can_transition_to(self):
        project = ResearchProject(
            title="Test SLR",
            topic="Testing systematic review transitions in the system",
            created_by="user_001",
        )
        assert project.can_transition_to("protocol_defined") is True
        assert project.can_transition_to("completed") is False

    def test_archived_has_no_transitions(self):
        project = ResearchProject(
            title="Archived SLR",
            topic="Testing systematic review archived status behavior",
            created_by="user_001",
            status="archived",
        )
        assert project.can_transition_to("draft") is False
        assert project.can_transition_to("archived") is False

    def test_title_min_length(self):
        with pytest.raises(Exception):
            ResearchProject(title="AB", topic="Topic " * 5, created_by="u")


# ============================================================
# ResearchProtocol
# ============================================================

class TestResearchProtocol:

    def _make_protocol(self) -> ResearchProtocol:
        return ResearchProtocol(
            research_id="research_001",
            research_questions=[
                ResearchQuestion(text="What ML methods are used for fraud detection?")
            ],
            inclusion_criteria=[
                "Studies published after 2015",
                "Studies using supervised learning",
            ],
            exclusion_criteria=[
                "Studies without evaluation metrics",
                "Studies not peer-reviewed",
            ],
        )

    def test_postconditions_valid(self):
        protocol = self._make_protocol()
        valid, errors = protocol.validate_postconditions()
        assert valid is True
        assert errors == []

    def test_postconditions_missing_rq(self):
        protocol = ResearchProtocol(
            research_id="r1",
            research_questions=[],
            inclusion_criteria=["c1", "c2"],
            exclusion_criteria=["e1", "e2"],
        )
        valid, errors = protocol.validate_postconditions()
        assert valid is False
        assert len(errors) >= 1

    def test_postconditions_insufficient_criteria(self):
        protocol = ResearchProtocol(
            research_id="r1",
            research_questions=[ResearchQuestion(text="Question de recherche numéro 1")],
            inclusion_criteria=["single_criterion"],
            exclusion_criteria=["e1", "e2"],
        )
        valid, errors = protocol.validate_postconditions()
        assert valid is False

    def test_mark_validated(self):
        protocol = self._make_protocol()
        assert not protocol.is_validated
        validated = protocol.mark_validated("reviewer_001")
        assert validated.is_validated
        assert validated.validated_by == "reviewer_001"

    def test_apply_modification_increments_version(self):
        protocol = self._make_protocol()
        assert protocol.version == 1
        modified = protocol.apply_modification(
            modified_by="user_001",
            field_changed="inclusion_criteria",
            old_value="Studies published after 2015",
            new_value="Studies published after 2010",
            justification="Élargissement de la plage temporelle pour inclure plus d'études",
        )
        assert modified.version == 2
        assert len(modified.modification_history) == 1
        assert modified.validated_by is None  # Revalidation requise

    def test_date_range_validation(self):
        with pytest.raises(ValueError):
            DateRange(start=2020, end=2015)  # start > end

    def test_date_range_valid(self):
        dr = DateRange(start=2015, end=2023)
        assert dr.start == 2015
        assert dr.end == 2023


# ============================================================
# SearchRun + SearchQuery
# ============================================================

class TestSearchRun:

    def test_total_results(self):
        run = SearchRun(
            research_id="r1",
            protocol_id="p1",
            protocol_version=1,
            results_by_source={"openalex": 42, "crossref": 18},
        )
        assert run.total_results == 60

    def test_total_after_dedup(self):
        run = SearchRun(
            research_id="r1",
            protocol_id="p1",
            protocol_version=1,
            results_by_source={"openalex": 100},
            duplicates_removed=15,
        )
        assert run.total_after_dedup == 85

    def test_has_results_false(self):
        run = SearchRun(
            research_id="r1",
            protocol_id="p1",
            protocol_version=1,
        )
        assert run.has_results is False

    def test_search_query_success(self):
        q = SearchQuery(
            search_run_id="sr1",
            source="openalex",
            query_string="fraud detection machine learning",
            http_status=200,
        )
        assert q.succeeded is True
        assert q.failed is False


# ============================================================
# Publication
# ============================================================

class TestPublication:

    def test_has_doi(self):
        pub = Publication(
            title="Deep Learning for Fraud Detection",
            source="openalex",
            doi="10.1234/test",
        )
        assert pub.has_doi is True

    def test_no_abstract(self):
        pub = Publication(title="Test Paper", source="crossref")
        assert pub.has_abstract is False

    def test_citation_key(self):
        pub = Publication(
            title="Deep Learning for Fraud Detection",
            source="openalex",
            authors=[Author(name="John Smith"), Author(name="Jane Doe")],
            year=2023,
        )
        assert pub.get_citation_key() == "Smith2023"


# ============================================================
# FullTextDocument
# ============================================================

class TestFullTextDocument:

    def test_initial_status(self):
        doc = FullTextDocument(
            publication_id="pub_001",
            file_path="data/raw/pdfs/paper.pdf",
            file_hash="abc123sha256",
        )
        assert doc.extraction_status == "pending"
        assert doc.needs_extraction is True
        assert doc.is_available is False

    def test_mark_extracted(self):
        doc = FullTextDocument(
            publication_id="pub_001",
            file_path="data/raw/pdfs/paper.pdf",
            file_hash="abc123sha256",
        )
        extracted = doc.mark_extracted(page_count=12, extractor_version="pymupdf-1.24.13")
        assert extracted.is_available is True
        assert extracted.page_count == 12

    def test_mark_failed(self):
        doc = FullTextDocument(
            publication_id="pub_001",
            file_path="data/raw/pdfs/paper.pdf",
            file_hash="abc123sha256",
        )
        failed = doc.mark_failed("PDF protégé par mot de passe")
        assert failed.extraction_status == "failed"
        assert "mot de passe" in failed.extraction_error


# ============================================================
# ScreeningDecision
# ============================================================

class TestScreeningDecision:

    def test_exclude_requires_reason(self):
        with pytest.raises(ValueError, match="exclusion_reason est obligatoire"):
            ScreeningDecision(
                publication_id="pub_001",
                research_id="r1",
                protocol_id="p1",
                screening_stage="title_abstract",
                decision="exclude",
                justification="Does not meet population criteria",
                agent="deterministic",
                confidence=0.95,
            )

    def test_exclude_with_valid_reason(self):
        decision = ScreeningDecision(
            publication_id="pub_001",
            research_id="r1",
            protocol_id="p1",
            screening_stage="title_abstract",
            decision="exclude",
            exclusion_reason="wrong_population",
            justification="Study does not focus on target population",
            agent="deterministic",
            confidence=0.95,
        )
        assert decision.decision == "exclude"

    def test_uncertain_low_confidence_requires_arbitration(self):
        decision = ScreeningDecision(
            publication_id="pub_001",
            research_id="r1",
            protocol_id="p1",
            screening_stage="title_abstract",
            decision="uncertain",
            justification="Unclear from title and abstract alone",
            agent="llm",
            confidence=0.45,
        )
        assert decision.requires_human_arbitration is True

    def test_uncertain_high_confidence_no_arbitration(self):
        decision = ScreeningDecision(
            publication_id="pub_001",
            research_id="r1",
            protocol_id="p1",
            screening_stage="title_abstract",
            decision="uncertain",
            justification="Needs closer examination of methodology",
            agent="llm",
            confidence=0.75,
        )
        assert decision.requires_human_arbitration is False

    def test_invalid_exclusion_reason(self):
        with pytest.raises(ValueError, match="Motif d'exclusion non standardisé"):
            ScreeningDecision(
                publication_id="pub_001",
                research_id="r1",
                protocol_id="p1",
                screening_stage="full_text",
                decision="exclude",
                exclusion_reason="bad_paper",  # Non standardisé
                justification="Not meeting quality criteria",
                agent="llm",
                confidence=0.8,
            )


# ============================================================
# Chunk
# ============================================================

class TestChunk:

    def test_reranking_eligible_text(self):
        chunk = Chunk(
            document_id="doc_001",
            publication_id="pub_001",
            text="The proposed method achieves an F1-score of 0.91 on the benchmark dataset.",
            chunk_index=0,
            char_start=0,
            char_end=74,
            embedding_model="all-MiniLM-L6-v2",
            embedding_version="v1",
        )
        assert chunk.is_semantic_reranking_eligible is True

    def test_table_not_reranking_eligible(self):
        chunk = Chunk(
            document_id="doc_001",
            publication_id="pub_001",
            text="| Method | Accuracy | F1 |\n| RF | 0.89 | 0.87 |",
            section_type="table",
            chunk_index=1,
            char_start=100,
            char_end=150,
            embedding_model="all-MiniLM-L6-v2",
            embedding_version="v1",
        )
        assert chunk.is_semantic_reranking_eligible is False

    def test_too_short(self):
        chunk = Chunk(
            document_id="doc_001",
            publication_id="pub_001",
            text="Short text",  # < 100 tokens estimés
            chunk_index=0,
            char_start=0,
            char_end=10,
            embedding_model="all-MiniLM-L6-v2",
            embedding_version="v1",
        )
        assert chunk.is_too_short is True


# ============================================================
# Evidence
# ============================================================

class TestEvidence:

    def _make_evidence(self, status: str = "extracted") -> Evidence:
        return Evidence(
            publication_id="pub_001",
            chunk_id="chunk_001",
            text="The Random Forest model achieved an F1-score of 0.91 on the Credit Card Fraud Dataset.",
            evidence_type="result",
            extracted_by_agent="research_assistant",
            extraction_confidence=0.9,
            verification_status=status,
            research_id="research_001",
        )

    def test_initial_not_trustworthy(self):
        ev = self._make_evidence("extracted")
        assert ev.is_trustworthy is False
        assert ev.is_pending is True

    def test_auto_verified_is_trustworthy(self):
        ev = self._make_evidence("auto_verified")
        assert ev.is_trustworthy is True

    def test_human_validated_is_trustworthy(self):
        ev = self._make_evidence("human_validated")
        assert ev.is_trustworthy is True

    def test_contested_requires_human(self):
        ev = self._make_evidence("contested")
        assert ev.requires_human_review is True

    def test_advance_to_auto_verified(self):
        ev = self._make_evidence("extracted")
        advanced = ev.advance_to(
            "auto_verified",
            verification_method="existence+provenance+semantic+entailment",
        )
        assert advanced.verification_status == "auto_verified"
        assert advanced.is_trustworthy is True


# ============================================================
# Claim + ClaimEvidenceLink
# ============================================================

class TestClaim:

    def test_initial_pending(self):
        claim = Claim(
            text="Random Forest outperforms SVM on fraud detection tasks.",
            rq_id="rq_001",
            report_version_id="rv_001",
            section="RQ1",
        )
        assert claim.status == "pending"
        assert claim.is_verified is False

    def test_supported_is_acceptable(self):
        claim = Claim(
            text="Random Forest outperforms SVM on fraud detection tasks.",
            rq_id="rq_001",
            report_version_id="rv_001",
            section="RQ1",
            status="supported",
        )
        assert claim.is_acceptable is True

    def test_contradicted_not_acceptable(self):
        claim = Claim(
            text="Deep learning is always superior to traditional ML.",
            rq_id="rq_001",
            report_version_id="rv_001",
            section="RQ1",
            status="contradicted",
        )
        assert claim.is_acceptable is False
        assert claim.needs_human_review is True

    def test_claim_evidence_link(self):
        link = ClaimEvidenceLink(
            claim_id="claim_001",
            evidence_id="ev_001",
            link_type="supports",
            link_confidence=0.92,
            created_by="auto",
        )
        assert link.link_type == "supports"


# ============================================================
# ReportVersion
# ============================================================

class TestReportVersion:

    def test_not_final_before_approval(self):
        report = ReportVersion(
            research_id="r1",
            version=1,
            content={"introduction": "...", "rq1": "..."},
            generated_by_agent="writing_agent",
        )
        assert report.is_final is False

    def test_mark_approved(self):
        report = ReportVersion(
            research_id="r1",
            version=1,
            content={"introduction": "...", "rq1": "..."},
            generated_by_agent="writing_agent",
        )
        approved = report.mark_approved("review_001")
        assert approved.is_final is True
        assert approved.human_review_id == "review_001"


# ============================================================
# AgentExecution
# ============================================================

class TestAgentExecution:

    def test_total_tokens(self):
        execution = AgentExecution(
            research_id="r1",
            agent_name="planning_agent",
            input_summary="Topic: fraud detection ML",
            tokens_input=500,
            tokens_output=200,
            llm_provider="ollama",
        )
        assert execution.total_tokens == 700

    def test_mark_completed(self):
        execution = AgentExecution(
            research_id="r1",
            agent_name="search_agent",
            input_summary="Protocol v1",
            llm_provider="ollama",
        )
        completed = execution.mark_completed(
            output_summary="Found 42 publications",
            duration_ms=3500,
        )
        assert completed.status == "completed"
        assert completed.succeeded is True

    def test_mark_failed(self):
        execution = AgentExecution(
            research_id="r1",
            agent_name="screening_agent",
            input_summary="100 publications to screen",
            llm_provider="groq",
        )
        failed = execution.mark_failed("API timeout after 60s", 60000)
        assert failed.status == "failed"
        assert failed.succeeded is False


# ============================================================
# HumanReview
# ============================================================

class TestHumanReview:

    def test_blocking_protocol_validation(self):
        review = HumanReview(
            research_id="r1",
            review_type="protocol_validation",
            reviewer_id="reviewer_001",
            decision="approved",
            justification="Protocol meets all scientific standards",
        )
        assert review.is_blocking is True
        assert review.is_protocol_validation is True

    def test_non_blocking_screening_arbitration(self):
        review = HumanReview(
            research_id="r1",
            review_type="screening_arbitration",
            reviewer_id="reviewer_001",
            decision="include",
            justification="Article is relevant to the research question",
        )
        assert review.is_blocking is False

    def test_report_approval_is_blocking(self):
        review = HumanReview(
            research_id="r1",
            review_type="report_approval",
            reviewer_id="reviewer_001",
            decision="approved",
            justification="All claims are properly supported by evidence",
        )
        assert review.is_blocking is True
        assert review.is_approval is True
