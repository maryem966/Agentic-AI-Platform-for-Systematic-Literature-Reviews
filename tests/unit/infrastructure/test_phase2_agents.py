from __future__ import annotations

from src.agents.search_agent import deduplicate_publications
from src.domain.entities.publication import Publication


def publication(title: str, doi: str | None, source: str) -> Publication:
    return Publication(title=title, doi=doi, source=source)


def test_deduplicate_uses_normalized_doi_and_title() -> None:
    results, removed = deduplicate_publications(
        [
            publication("First paper", "10.1234/ABC", "openalex"),
            publication("Different title", "10.1234/abc", "crossref"),
            publication("  A Study: On RAG! ", None, "openalex"),
            publication("a study on rag", None, "semantic_scholar"),
        ]
    )
    assert len(results) == 2
    assert removed == 2
