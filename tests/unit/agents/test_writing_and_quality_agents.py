"""
Tests unitaires pour le WritingAgent et le QualityAgent.
Vérifie la couverture RQ, le grounding, la réflexion bornée et l'ancrage des claims.
"""
from __future__ import annotations

import pytest

from src.agents.quality_agent import QualityAgent
from src.agents.writing_agent import WritingAgent, WritingSynthesisOutput, ReportSectionDraft, ClaimDraft
from src.domain.entities.claim import Claim, ClaimEvidenceLink
from src.domain.entities.evidence import Evidence
from src.domain.entities.report_version import ReportVersion
from src.domain.entities.research_protocol import ResearchQuestion
from src.domain.ports.llm_port import LLMPort, LLMResponse


class MockWritingLLM(LLMPort):
    """Simulateur de LLM pour le WritingAgent."""

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-writer"

    async def health_check(self) -> bool:
        return True

    async def invoke(self, messages, temperature=0.0, max_tokens=None):
        return LLMResponse(content="", model=self.model_name, provider=self.provider_name)

    async def invoke_structured(self, messages, response_model, temperature=0.0, max_repair_attempts=2):
        # Renvoie une synthèse structurée avec un Claim ancré et un Claim non ancré
        return WritingSynthesisOutput(
            executive_summary="Synthèse exécutive basée sur la littérature analysée.",
            general_conclusion="Les résultats confirment l'efficacité des méthodes supervisées.",
            sections=[
                ReportSectionDraft(
                    section_name="RQ1_Methods",
                    rq_id="rq1",
                    content="L'analyse des études montre que Random Forest domine avec un F1-score de 0.91.",
                    claims=[
                        ClaimDraft(
                            text="Random Forest atteint un score de 0.91 sur le dataset CreditCard.",
                            rq_id="rq1",
                            section="RQ1_Methods",
                            cited_evidence_ids=["ev_001"],
                        )
                    ],
                )
            ],
        )


class MockQualityLLM(LLMPort):
    """Simulateur de LLM-as-a-Judge."""

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-judge"

    async def health_check(self) -> bool:
        return True

    async def invoke(self, messages, temperature=0.0, max_tokens=None):
        return LLMResponse(content="", model=self.model_name, provider=self.provider_name)

    async def invoke_structured(self, messages, response_model, temperature=0.0, max_repair_attempts=2):
        from src.agents.quality_agent import LLMJudgeReportEvaluation
        return LLMJudgeReportEvaluation(
            coherence_score=0.90,
            scientific_rigor_score=0.88,
            critique_findings=[],
            recommended_verdict="ACCEPT",
        )


@pytest.mark.asyncio
async def test_writing_agent_generates_grounded_report():
    llm = MockWritingLLM()
    agent = WritingAgent(llm)

    questions = [ResearchQuestion(id="rq1", text="Quelles sont les méthodes de détection ?")]
    evidences = [
        Evidence(
            id="ev_001",
            publication_id="pub1",
            chunk_id="chunk1",
            text="Random Forest achieved 0.91 F1-score on CreditCard dataset.",
            evidence_type="result",
            extracted_by_agent="assistant",
            extraction_confidence=0.95,
            verification_status="auto_verified",
            research_id="res1",
        )
    ]

    report, claims, links = await agent.write_report("res1", questions, evidences)

    assert report.version == 1
    assert "RQ1_Methods" in report.content
    assert len(claims) == 1
    assert claims[0].status == "supported"
    assert len(links) == 1
    assert links[0].evidence_id == "ev_001"


@pytest.mark.asyncio
async def test_quality_agent_computes_coverage_and_grounding():
    quality_llm = MockQualityLLM()
    agent = QualityAgent(quality_llm)

    questions = [
        ResearchQuestion(id="rq1", text="Méthodes ?"),
        ResearchQuestion(id="rq2", text="Métriques ?"),
    ]
    report = ReportVersion(
        id="rep1",
        research_id="res1",
        version=1,
        content={"RQ1_Methods": "Texte couvrant RQ1"},
        generated_by_agent="writing_agent",
    )
    claims = [
        Claim(id="c1", text="Claim 1 - Random Forest achieves high F1", rq_id="rq1", report_version_id="rep1", section="RQ1_Methods", status="supported"),
        Claim(id="c2", text="Claim 2 orphelin sans preuve valable", rq_id="rq1", report_version_id="rep1", section="RQ1_Methods", status="unsupported"),
    ]
    evidences = [
        Evidence(
            id="ev1", publication_id="p1", chunk_id="chk1", text="Preuve", evidence_type="result",
            extracted_by_agent="assistant", extraction_confidence=0.9, verification_status="auto_verified", research_id="res1"
        )
    ]
    links = [ClaimEvidenceLink(id="l1", claim_id="c1", evidence_id="ev1", link_type="supports", link_confidence=0.9, created_by="auto")]

    # RQ1 est couverte, RQ2 manque -> couverture = 50%
    metrics = agent.compute_deterministic_metrics(report, questions, claims, links, evidences)
    assert metrics["rq_coverage"] == 0.5
    assert "rq2" in metrics["missing_rqs"]
    assert metrics["grounding_ratio"] == 0.5
    assert "c2" in metrics["unsupported_claims"]


@pytest.mark.asyncio
async def test_quality_agent_verdict_triggers_revision_and_escalate():
    quality_llm = MockQualityLLM()
    agent = QualityAgent(quality_llm)

    questions = [ResearchQuestion(id="rq1", text="Méthodes ?")]
    report = ReportVersion(id="rep1", research_id="res1", version=1, content={}, generated_by_agent="writer")
    claims = [Claim(id="c1", text="Non prouvé", rq_id="rq1", report_version_id="rep1", section="sec", status="unsupported")]

    # 1. Première passe (revision_count = 0) -> REVISE
    verdict_1 = await agent.evaluate_report(report, questions, claims, [], [], revision_count=0, max_revisions=2)
    assert verdict_1.verdict == "REVISE"
    assert verdict_1.repair_target in ("writing", "retrieval")

    # 2. Budget épuisé (revision_count = 2) -> ESCALATE
    verdict_2 = await agent.evaluate_report(report, questions, claims, [], [], revision_count=2, max_revisions=2)
    assert verdict_2.verdict == "ESCALATE"
    assert verdict_2.repair_target == "human"
