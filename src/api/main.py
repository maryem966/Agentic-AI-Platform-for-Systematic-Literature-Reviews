"""SLR-Nexus API and Phase 2 workflow endpoints."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from ..agents.search_agent import SearchAgent
from ..application.dto.research_dtos import RAGQueryDTO
from ..config.settings import settings
from ..infrastructure.persistence.repositories import ProjectRepo, ScreeningRepo
from ..infrastructure.scientific_apis.crossref_adapter import CrossrefAdapter
from ..infrastructure.scientific_apis.openalex_adapter import OpenAlexAdapter
from ..infrastructure.scientific_apis.semantic_scholar_adapter import SemanticScholarAdapter
from ..infrastructure.vector.qdrant_adapter import QdrantVectorStoreAdapter
from ..orchestration.postgres_checkpointer import (
    PostgreSQLCheckpointer,
    configure_postgres_event_loop,
)
from ..orchestration.workflow import build_research_graph
from ..rag.service import RAGService

configure_postgres_event_loop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = PostgreSQLCheckpointer()
    saver = await checkpointer.open()
    session_factory = create_session_factory()
    adapters = {
        "openalex": OpenAlexAdapter(),
        "crossref": CrossrefAdapter(),
        "semantic_scholar": SemanticScholarAdapter(),
    }
    # A missing optional Groq key must not prevent use of the local Ollama provider.
    llm = create_default_llm_adapter() if settings.groq_api_key else OllamaAdapter()
    vector_store = QdrantVectorStoreAdapter()
    app.state.session_factory = session_factory
    app.state.llm = llm
    app.state.vector_store = vector_store
    app.state.adapters = adapters
    app.state.graph = build_research_graph(
        llm=llm,
        session_factory=session_factory,
        search_agent=SearchAgent(adapters),
        checkpointer=saver,
        vector_store=vector_store,
    )
    app.state.checkpointer = checkpointer
    try:
        yield
    finally:
        for adapter in adapters.values():
            await adapter.close()
        await vector_store.client.close()
        engine = session_factory.kw.get("bind")
        if engine is not None:
            await engine.dispose()
        await checkpointer.close()


app = FastAPI(title="SLR-Nexus", version="0.2.0", lifespan=lifespan)


class StartResearchBody(BaseModel):
    objectives: list[str] = Field(default_factory=list)
    requested_by: str = "user"


class ValidationBody(BaseModel):
    reviewer_id: str = Field(min_length=1)
    decision: str
    justification: str = Field(default="", min_length=5, max_length=4000)
    exclusion_reason: str | None = None


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Expose l'état du processus sans dépendre des services externes."""
    return {"status": "ok"}


@app.get("/api/v1/research/{research_id}/prisma", tags=["research"])
async def get_prisma_flow(research_id: str) -> Any:
    """Calculate the PRISMA flow directly from PostgreSQL records."""
    async with app.state.session_factory() as session:
        flow = await ScreeningRepo(session).calculate_prisma_flow(research_id)
    return jsonable_encoder(flow)


@app.post("/api/v1/research/{research_id}/rag", tags=["research"])
async def query_research_rag(research_id: str, body: RAGQueryDTO) -> Any:
    """Retrieve evidence from included publications and return chunk-level citations."""
    if body.research_id != research_id:
        raise HTTPException(status_code=422, detail="research_id du corps et du chemin différents")
    try:
        async with app.state.session_factory() as session:
            response = await RAGService(app.state.vector_store, app.state.llm).query(
                session,
                research_id=research_id,
                question=body.query,
                top_k=body.top_k,
                strategy=body.strategy,
                route=body.route,
            )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return jsonable_encoder(response)


@app.post("/api/v1/research/{research_id}/run", tags=["research"])
async def start_research(research_id: str, body: StartResearchBody) -> Any:
    """Start planning; the protocol review suspends until the human validates it."""
    async with app.state.session_factory() as session:
        project = await ProjectRepo(session).find_by_id(research_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Projet de recherche introuvable")
    state = {
        "research_id": research_id,
        "thread_id": research_id,
        "topic": project.topic,
        "objectives": body.objectives,
        "requested_by": body.requested_by,
    }
    result = await app.state.graph.ainvoke(state, {"configurable": {"thread_id": research_id}})
    return jsonable_encoder(result)


@app.post("/api/v1/research/{research_id}/validate", tags=["research"])
async def validate_research(research_id: str, body: ValidationBody) -> Any:
    """Resume a paused workflow with an explicit human decision."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": research_id}}
    snapshot = await app.state.graph.aget_state(config)
    pending = [item for task in snapshot.tasks for item in task.interrupts]
    if not pending:
        raise HTTPException(status_code=409, detail="Aucune validation humaine en attente")
    allowed = set(pending[0].value.get("required_decisions", []))
    decision = next(
        (choice for choice in allowed if choice.casefold() == body.decision.casefold()), None
    )
    if decision is None:
        raise HTTPException(
            status_code=422,
            detail=f"Décision non autorisée pour cette étape; choix: {sorted(allowed)}",
        )
    if decision.casefold() == "exclude":
        from ..domain.entities.screening_decision import STANDARD_EXCLUSION_REASONS

        if body.exclusion_reason not in STANDARD_EXCLUSION_REASONS:
            raise HTTPException(
                status_code=422,
                detail=f"exclusion_reason doit appartenir à {sorted(STANDARD_EXCLUSION_REASONS)}",
            )
    result = await app.state.graph.ainvoke(
        Command(
            resume={
                "decision": decision,
                "reviewer_id": body.reviewer_id,
                "justification": body.justification,
                "exclusion_reason": body.exclusion_reason,
            }
        ),
        config,
    )
    return jsonable_encoder(result)
