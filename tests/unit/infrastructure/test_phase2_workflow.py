from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.orchestration.workflow import build_research_graph


def test_workflow_compiles_with_review_and_phase_nodes() -> None:
    graph = build_research_graph(
        llm=None,
        session_factory=None,
        search_agent=None,
        checkpointer=MemorySaver(),
    )
    nodes = set(graph.get_graph().nodes)
    assert {
        "planning",
        "protocol_review",
        "search",
        "screening",
        "document_indexing",
        "evidence_extraction",
        "writing",
        "critic",
        "quality",
    }.issubset(nodes)


class ApprovalState(TypedDict, total=False):
    decision: str


def test_langgraph_interrupt_can_resume_from_checkpoint() -> None:
    def approval(_state: ApprovalState) -> dict[str, str]:
        return {"decision": interrupt({"kind": "protocol_validation"})}

    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "resume-test"}}

    events = list(graph.stream({"decision": ""}, config))
    interrupted = next(event["__interrupt__"] for event in events if "__interrupt__" in event)
    assert interrupted[0].value["kind"] == "protocol_validation"
    resumed = graph.invoke(Command(resume="ACCEPT"), config)
    assert resumed["decision"] == "ACCEPT"
