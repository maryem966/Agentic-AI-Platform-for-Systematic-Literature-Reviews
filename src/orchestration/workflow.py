"""LangGraph workflow skeleton; external effects stay behind adapters and repositories."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..agents.planning_agent import PlanningAgent
from ..agents.research_assistant_agent import ResearchAssistantAgent
from ..agents.screening_agent import ScreeningAgent
from ..agents.search_agent import SearchAgent
from ..application.dto.research_dtos import PlanningInput
from ..domain.entities.human_review import HumanReview
from ..domain.entities.publication import Publication
from ..domain.entities.research_protocol import ResearchProtocol
from ..evidence.registry import EvidenceRegistry
from ..evidence.verifier import FourLevelClaimVerifier
from ..infrastructure.persistence.repositories import (
    ChunkRepo,
    EvidenceRepo,
    ExtractionRecordRepo,
    FullTextDocumentRepo,
    ProjectRepo,
    ProtocolRepo,
    PublicationRepo,
    ScreeningRepo,
    SearchRunRepo,
)
from ..rag.document_ingestion import PDFIngestion
from ..rag.indexing import DocumentIndexingPipeline
from .state import ResearchState


def build_research_graph(
    *,
    llm: Any,
    session_factory: Any,
    search_agent: SearchAgent,
    checkpointer: Any,
    vector_store: Any | None = None,
) -> Any:
    """Build the resumable graph. The saver is supplied by the application lifespan."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    planning_agent = PlanningAgent(llm)
    screening_agent = ScreeningAgent(llm)
    assistant_agent = ResearchAssistantAgent(llm)
    indexing_pipeline = DocumentIndexingPipeline(vector_store) if vector_store else None

    async def planning(state: ResearchState) -> dict[str, Any]:
        objectives = list(state.get("objectives", []))
        feedback = state.get("review_justification")
        if feedback:
            objectives.append(f"Retour humain à intégrer dans la révision: {feedback}")
        request = PlanningInput(
            research_id=state["research_id"],
            topic=state["topic"],
            objectives=objectives,
            requested_by=state.get("requested_by", "system"),
        )
        async with session_factory() as session:
            project = await ProjectRepo(session).find_by_id(request.research_id)
            if project is None:
                raise ValueError("Projet de recherche introuvable")
            latest = await ProtocolRepo(session).find_latest_by_research(request.research_id)
            protocol = await planning_agent.plan(request)
            revision = state.get("revision_count", 0)
            protocol_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"slr-nexus:protocol:{request.research_id}:{state.get('thread_id', request.research_id)}:{revision}",
                )
            )
            version = (
                latest.version
                if latest and latest.id == protocol_id
                else (latest.version + 1 if latest else 1)
            )
            protocol = protocol.model_copy(update={"id": protocol_id, "version": version})
            await ProtocolRepo(session).save(protocol)
            await session.commit()
        return {"protocol": protocol.model_dump(mode="json"), "protocol_id": protocol.id}

    async def protocol_review(state: ResearchState) -> dict[str, Any]:
        response = interrupt(
            {
                "kind": "protocol_validation",
                "research_id": state["research_id"],
                "protocol": state["protocol"],
                "required_decisions": ["ACCEPT", "REVISE", "ESCALATE"],
            }
        )
        if not isinstance(response, dict):
            raise ValueError("La reprise doit fournir une décision humaine structurée")
        decision = str(response.get("decision", "")).upper()
        if decision not in {"ACCEPT", "REVISE", "ESCALATE"}:
            raise ValueError("Décision invalide; valeurs autorisées: ACCEPT, REVISE, ESCALATE")
        async with session_factory() as session:
            reviewer_id = str(response.get("reviewer_id", "human"))
            justification = str(response.get("justification", ""))
            review = HumanReview(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"slr-nexus:review:protocol:{state['protocol_id']}:{decision}:{reviewer_id}",
                    )
                ),
                research_id=state["research_id"],
                review_type="protocol_validation",
                reviewer_id=reviewer_id,
                context_snapshot={"protocol": state["protocol"]},
                decision=decision,
                justification=justification or "Décision de validation du protocole.",
            )
            from ..infrastructure.persistence.models import HumanReviewORM

            review_row = await session.get(HumanReviewORM, review.id)
            if review_row is None:
                session.add(HumanReviewORM(**review.model_dump(mode="python")))
            if decision == "ACCEPT":
                protocol = ResearchProtocol.model_validate(state["protocol"])
                await ProtocolRepo(session).save(protocol.mark_validated(reviewer_id))
            await session.commit()
        return {
            "protocol_decision": decision,
            "reviewer_id": reviewer_id,
            "review_justification": justification,
            "revision_count": state.get("revision_count", 0) + (1 if decision == "REVISE" else 0),
        }

    async def search(state: ResearchState) -> dict[str, Any]:
        protocol = ResearchProtocol.model_validate(state["protocol"])
        run, publications = await search_agent.search(state["research_id"], protocol)
        async with session_factory() as session:
            await SearchRunRepo(session).save(run)
            await PublicationRepo(session).bulk_save(publications)
            await session.commit()
        return {
            "search_run_id": run.id,
            "search_results": [item.model_dump(mode="json") for item in publications],
            "search_summary": {
                "results_by_source": run.results_by_source,
                "duplicates_removed": run.duplicates_removed,
                "unique_publications": len(publications),
            },
            "workflow_decision": "ACCEPT" if publications else "ESCALATE",
        }

    async def screening(state: ResearchState) -> dict[str, Any]:
        protocol = ResearchProtocol.model_validate(state["protocol"])
        decisions = []
        reviews: list[HumanReview] = []
        for raw in state.get("search_results", []):
            publication = Publication.model_validate(raw)
            decision = await screening_agent.screen(publication, protocol)
            review_row = None
            if decision.requires_human_arbitration:
                response = interrupt(
                    {
                        "kind": "screening_arbitration",
                        "research_id": state["research_id"],
                        "publication": publication.model_dump(mode="json"),
                        "decision": decision.model_dump(mode="json"),
                        "required_decisions": ["include", "exclude"],
                    }
                )
                if not isinstance(response, dict) or response.get("decision") not in {
                    "include",
                    "exclude",
                }:
                    raise ValueError(
                        "La reprise du screening doit contenir une décision humaine valide"
                    )
                if response["decision"] == "exclude" and not response.get("exclusion_reason"):
                    raise ValueError("Une exclusion humaine exige un motif standardisé")
                reviewer_id = str(response.get("reviewer_id", "human"))
                review_row = HumanReview(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"slr-nexus:screen-review:{protocol.id}:{publication.id}:{reviewer_id}",
                        )
                    ),
                    research_id=state["research_id"],
                    review_type="screening_arbitration",
                    reviewer_id=reviewer_id,
                    context_snapshot={
                        "publication_id": publication.id,
                        "automated_decision": decision.model_dump(mode="json"),
                    },
                    decision=response["decision"],
                    justification=str(
                        response.get("justification", "Arbitrage humain du screening.")
                    ),
                )
                reviews.append(review_row)
                decision = decision.override_with_human_decision(
                    response["decision"], review_row.id, response.get("exclusion_reason")
                )
            decision = decision.model_copy(
                update={
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"slr-nexus:screen:{protocol.id}:{publication.id}:title_abstract",
                        )
                    )
                }
            )
            decisions.append(decision)
        async with session_factory() as session:
            repo = ScreeningRepo(session)
            from ..infrastructure.persistence.models import HumanReviewORM

            for review in reviews:
                if await session.get(HumanReviewORM, review.id) is None:
                    session.add(HumanReviewORM(**review.model_dump(mode="python")))
            for decision in decisions:
                await repo.save(decision)
            await session.commit()
        return {
            "screening_decisions": [decision.model_dump(mode="json") for decision in decisions],
            "workflow_decision": "ACCEPT",
        }

    async def document_indexing(state: ResearchState) -> dict[str, Any]:
        if indexing_pipeline is None:
            return {"indexed_publications": [], "workflow_decision": "ACCEPT"}
        protocol = ResearchProtocol.model_validate(state["protocol"])
        accepted = [
            item
            for item in state.get("screening_decisions", [])
            if item.get("decision") == "include"
        ]
        results = []
        full_text_decisions = []
        reviews: list[HumanReview] = []
        pdf_ingestion = PDFIngestion()
        async with session_factory() as session:
            by_id: dict[str, Publication] = {}
            for raw in state.get("search_results", []):
                item = Publication.model_validate(raw)
                by_id[item.id] = item
            for screening_result in accepted:
                publication = by_id.get(screening_result["publication_id"])
                if not publication:
                    continue
                document_url = publication.url or (
                    f"https://doi.org/{publication.doi}" if publication.doi else None
                )
                chunks = (
                    await indexing_pipeline.index(
                        session, publication, document_url, Path("data/raw/pdfs")
                    )
                    if document_url
                    else []
                )
                if not document_url:
                    await indexing_pipeline._record_failure(
                        session,
                        publication,
                        ValueError("Aucune URL ou DOI disponible pour le texte intégral"),
                        inaccessible=True,
                    )
                text = None
                source_excerpt = None
                injection_flag = False
                extraction_status = "inaccessible"
                if publication.document_id:
                    document = await FullTextDocumentRepo(session).find_by_id(
                        publication.document_id
                    )
                    if document:
                        extraction_status = document.extraction_status
                    if document and document.file_path:
                        extracted = pdf_ingestion.extract(Path(document.file_path).read_bytes())
                        injection_flag = extracted.prompt_injection_suspected
                        full_document_text = "\n".join(
                            page_text for _, page_text in extracted.pages
                        )
                        source_excerpt = full_document_text[:8000]
                        if not injection_flag:
                            text = full_document_text
                full_decision = await screening_agent.screen(
                    publication, protocol, stage="full_text", full_text=text
                )
                review = None
                if full_decision.requires_human_arbitration:
                    response = interrupt(
                        {
                            "kind": "screening_arbitration",
                            "research_id": state["research_id"],
                            "publication": publication.model_dump(mode="json"),
                            "decision": full_decision.model_dump(mode="json"),
                            "document_excerpt_untrusted": source_excerpt,
                            "prompt_injection_suspected": injection_flag,
                            "required_decisions": ["include", "exclude"],
                        }
                    )
                    if not isinstance(response, dict) or response.get("decision") not in {
                        "include",
                        "exclude",
                    }:
                        raise ValueError(
                            "La reprise du screening intégral exige include ou exclude"
                        )
                    if response["decision"] == "exclude" and not response.get("exclusion_reason"):
                        raise ValueError("Une exclusion humaine exige un motif standardisé")
                    reviewer_id = str(response.get("reviewer_id", "human"))
                    review = HumanReview(
                        id=str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"slr-nexus:fullscreen-review:{protocol.id}:{publication.id}:{reviewer_id}",
                            )
                        ),
                        research_id=state["research_id"],
                        review_type="screening_arbitration",
                        reviewer_id=reviewer_id,
                        context_snapshot={
                            "publication_id": publication.id,
                            "automated_decision": full_decision.model_dump(mode="json"),
                        },
                        decision=response["decision"],
                        justification=str(
                            response.get("justification", "Arbitrage humain du texte intégral.")
                        ),
                    )
                    reviews.append(review)
                    full_decision = full_decision.override_with_human_decision(
                        response["decision"], review.id, response.get("exclusion_reason")
                    )
                full_decision = full_decision.model_copy(
                    update={
                        "id": str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"slr-nexus:screen:{protocol.id}:{publication.id}:full_text",
                            )
                        )
                    }
                )
                full_text_decisions.append(full_decision)
                results.append(
                    {
                        "publication_id": publication.id,
                        "status": extraction_status,
                        "chunks": len(chunks),
                        "prompt_injection_suspected": injection_flag,
                    }
                )
            from ..infrastructure.persistence.models import HumanReviewORM

            for review in reviews:
                if await session.get(HumanReviewORM, review.id) is None:
                    session.add(HumanReviewORM(**review.model_dump(mode="python")))
            for full_text_decision in full_text_decisions:
                await ScreeningRepo(session).save(full_text_decision)
            await session.commit()
        return {
            "indexed_publications": results,
            "screening_decisions": [item.model_dump(mode="json") for item in full_text_decisions],
            "workflow_decision": "ACCEPT",
        }

    async def evidence_extraction(state: ResearchState) -> dict[str, Any]:
        research_id = state["research_id"]
        protocol_dict = state.get("protocol") or {}
        report_version_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"slr-nexus:report-version:{research_id}:v1",
            )
        )
        extracted_records = []
        claims_data = []

        async with session_factory() as session:
            protocol = await ProtocolRepo(session).find_latest_by_research(research_id)
            if protocol is None and protocol_dict:
                protocol = ResearchProtocol.model_validate(protocol_dict)

            decisions = await ScreeningRepo(session).find_by_research(research_id)
            included_pub_ids = {
                d.publication_id
                for d in decisions
                if d.decision == "include"
            }
            if not included_pub_ids:
                publications = await PublicationRepo(session).find_by_research(research_id)
                included_pub_ids = {p.id for p in publications}

            valid_pubs = []
            for pid in included_pub_ids:
                pub = await PublicationRepo(session).find_by_id(pid)
                if pub is not None:
                    valid_pubs.append(pub)

            evidence_repo = EvidenceRepo(session)
            registry = EvidenceRegistry(evidence_repo, session)
            extraction_repo = ExtractionRecordRepo(session)
            chunk_repo = ChunkRepo(session)

            all_evidences = []

            for pub in valid_pubs:
                chunks = await chunk_repo.find_by_publication(pub.id)
                if not chunks:
                    continue
                record, evs = await assistant_agent.extract_from_publication(
                    pub,
                    chunks,
                    research_id=research_id,
                    agent_execution_id=str(uuid.uuid4()),
                )
                saved_record = await extraction_repo.save(record)
                extracted_records.append(saved_record)

                for ev in evs:
                    try:
                        saved_ev = await registry.create(ev)
                        all_evidences.append(saved_ev)
                    except Exception:
                        pass

            if not all_evidences:
                all_evidences = await evidence_repo.find_evidence_by_research(research_id)

            verifier = FourLevelClaimVerifier(
                evidence_port=evidence_repo,
                chunk_resolver=lambda cid: chunk_repo.find_by_id(cid),
                llm=llm,
                similarity_threshold=0.50,
            )

            if all_evidences and protocol:
                claims_with_links = await assistant_agent.formulate_claims(
                    research_id=research_id,
                    report_version_id=report_version_id,
                    research_protocol=protocol,
                    evidences=all_evidences,
                )

                for claim, links in claims_with_links:
                    claim_evs = [
                        ev for ev in all_evidences
                        if ev.id in {lnk.evidence_id for lnk in links}
                    ]
                    verification = await verifier.verify(claim, claim_evs)
                    claim = claim.model_copy(
                        update={
                            "status": verification.status,
                            "verification_method": verification.verification_method,
                            "contradicting_evidence_ids": verification.contradicting_evidence_ids,
                        }
                    )
                    await evidence_repo.save_claim(claim)
                    for lnk in links:
                        await evidence_repo.link_claim_to_evidence(lnk)

                    for ev in claim_evs:
                        new_status = "auto_verified" if verification.is_acceptable else "auto_rejected"
                        try:
                            await registry.update_status(
                                ev.id,
                                new_status,
                                verification_method=verification.verification_method,
                            )
                        except Exception:
                            pass

                    claims_data.append(claim.model_dump(mode="json"))

            await session.commit()

        return {
            "extractions": [r.model_dump(mode="json") for r in extracted_records],
            "claims": claims_data,
            "workflow_decision": "ACCEPT",
        }

    async def writing(state: ResearchState) -> dict[str, Any]:
        return {"workflow_decision": "ACCEPT"}

    async def critic(state: ResearchState) -> dict[str, Any]:
        return {"workflow_decision": "ACCEPT"}

    async def quality(state: ResearchState) -> dict[str, Any]:
        return {"workflow_decision": "ACCEPT"}

    def after_review(state: ResearchState) -> str:
        return state["protocol_decision"].lower()

    def after_search(state: ResearchState) -> str:
        return "escalate" if not state.get("search_results") else "screening"

    def after_quality(state: ResearchState) -> str:
        decision = state.get("workflow_decision", "ESCALATE").lower()
        return decision if decision in {"accept", "revise", "escalate"} else "escalate"

    async def search_escalation(state: ResearchState) -> dict[str, Any]:
        response = interrupt(
            {
                "kind": "zero_search_results",
                "research_id": state["research_id"],
                "search_run_id": state.get("search_run_id"),
                "message": "Aucun résultat trouvé. Vérifier le protocole ou les requêtes.",
                "required_decisions": ["REVISE", "ESCALATE"],
            }
        )
        decision = (
            str(response.get("decision", "ESCALATE")).upper()
            if isinstance(response, dict)
            else "ESCALATE"
        )
        if decision not in {"REVISE", "ESCALATE"}:
            raise ValueError("Après zéro résultat, seule une révision ou une escalade est permise")
        if isinstance(response, dict):
            from ..infrastructure.persistence.models import HumanReviewORM

            review = HumanReview(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"slr-nexus:review:empty-search:{state['search_run_id']}:{decision}:{response.get('reviewer_id', 'human')}",
                    )
                ),
                research_id=state["research_id"],
                review_type="quality_escalation",
                reviewer_id=str(response.get("reviewer_id", "human")),
                context_snapshot={"search_run_id": state.get("search_run_id")},
                decision=decision,
                justification=str(response.get("justification", "Décision après zéro résultat.")),
            )
            async with session_factory() as session:
                review_row = await session.get(HumanReviewORM, review.id)
                if review_row is None:
                    session.add(HumanReviewORM(**review.model_dump(mode="python")))
                await session.commit()
        return {
            "workflow_decision": decision,
            "revision_count": state.get("revision_count", 0) + (1 if decision == "REVISE" else 0),
            "review_justification": str(response.get("justification", ""))
            if isinstance(response, dict)
            else "",
        }

    def after_search_escalation(state: ResearchState) -> str:
        if state.get("workflow_decision") == "REVISE":
            return "planning"
        return "end"

    builder = StateGraph(ResearchState)
    for name, node in (
        ("planning", planning),
        ("protocol_review", protocol_review),
        ("search", search),
        ("search_escalation", search_escalation),
        ("screening", screening),
        ("document_indexing", document_indexing),
        ("evidence_extraction", evidence_extraction),
        ("writing", writing),
        ("critic", critic),
        ("quality", quality),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "planning")
    builder.add_edge("planning", "protocol_review")
    builder.add_conditional_edges(
        "protocol_review",
        after_review,
        {
            "accept": "search",
            "revise": "planning",
            "escalate": END,
        },
    )
    builder.add_conditional_edges(
        "search",
        after_search,
        {
            "screening": "screening",
            "escalate": "search_escalation",
        },
    )
    builder.add_edge("screening", "document_indexing")
    builder.add_edge("document_indexing", "evidence_extraction")
    builder.add_edge("evidence_extraction", "writing")
    builder.add_edge("writing", "critic")
    builder.add_edge("critic", "quality")
    builder.add_conditional_edges(
        "quality",
        after_quality,
        {
            "accept": END,
            "revise": "writing",
            "escalate": END,
        },
    )
    builder.add_conditional_edges(
        "search_escalation", after_search_escalation, {"planning": "planning", "end": END}
    )
    return builder.compile(checkpointer=checkpointer)
