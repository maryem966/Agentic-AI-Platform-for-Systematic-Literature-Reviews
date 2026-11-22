"""Shared, serializable state for the SLR workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    research_id: str
    thread_id: str
    topic: str
    objectives: list[str]
    requested_by: str
    protocol: dict[str, Any]
    protocol_id: str
    revision_count: int
    protocol_decision: str
    reviewer_id: str
    review_justification: str
    search_run_id: str
    search_results: list[dict[str, Any]]
    search_summary: dict[str, Any]
    screening_decisions: list[dict[str, Any]]
    indexed_publications: list[dict[str, Any]]
    extractions: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    workflow_decision: str
    repair_target: str
    error: str | None
