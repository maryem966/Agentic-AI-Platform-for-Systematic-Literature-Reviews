"""Deterministic, auditable search over the three scientific APIs."""

from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any, Literal, cast

from ..domain.entities.publication import Publication
from ..domain.entities.research_protocol import ResearchProtocol
from ..domain.entities.search_run import SearchQuery, SearchRun

SearchAdapter = Any
SearchSource = Literal["openalex", "crossref", "semantic_scholar", "manual"]


def _normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def deduplicate_publications(publications: list[Publication]) -> tuple[list[Publication], int]:
    """Deduplicate first by normalized DOI, then by normalized title."""
    unique: dict[str, Publication] = {}
    for publication in publications:
        doi = (
            (publication.doi or "")
            .strip()
            .lower()
            .removeprefix("https://doi.org/")
            .removeprefix("http://doi.org/")
        )
        key = f"doi:{doi}" if doi else f"title:{_normalize_title(publication.title)}"
        current = unique.get(key)
        if current is None or (not current.abstract and publication.abstract):
            unique[key] = publication
    result = list(unique.values())
    return result, len(publications) - len(result)


class SearchAgent:
    def __init__(self, adapters: dict[str, SearchAdapter]) -> None:
        self.adapters = adapters

    async def search(
        self, research_id: str, protocol: ResearchProtocol, limit: int = 25
    ) -> tuple[SearchRun, list[Publication]]:
        queries: list[SearchQuery] = []
        all_results: list[Publication] = []
        counts = {source: 0 for source in protocol.target_databases}
        run_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"slr-nexus:search:{research_id}:{protocol.id}")
        )
        start = protocol.date_range.start if protocol.date_range else None
        end = protocol.date_range.end if protocol.date_range else None
        for question in protocol.research_questions:
            for source in protocol.target_databases:
                adapter = self.adapters.get(source)
                if adapter is None:
                    raise ValueError(f"Source scientifique inconnue: {source}")
                if source not in {"openalex", "crossref", "semantic_scholar"}:
                    raise ValueError(f"Source scientifique non prise en charge: {source}")
                search_source = cast(SearchSource, source)
                exact_query = question.text
                filters = {"start_year": start, "end_year": end, "limit": limit}
                try:
                    found = await adapter.search(
                        exact_query, start_year=start, end_year=end, limit=limit
                    )
                    queries.append(
                        SearchQuery(
                            id=str(uuid.uuid5(uuid.UUID(run_id), f"{source}:{question.id}")),
                            search_run_id=run_id,
                            source=search_source,
                            query_string=exact_query,
                            filters_applied=filters,
                            results_count=len(found),
                            http_status=200,
                        )
                    )
                    counts[source] += len(found)
                    all_results.extend(found)
                except Exception as exc:
                    queries.append(
                        SearchQuery(
                            id=str(uuid.uuid5(uuid.UUID(run_id), f"{source}:{question.id}")),
                            search_run_id=run_id,
                            source=search_source,
                            query_string=exact_query,
                            filters_applied=filters,
                            error_message=f"{type(exc).__name__}: {exc}",
                        )
                    )
        unique, removed = deduplicate_publications(all_results)
        run = SearchRun(
            id=run_id,
            research_id=research_id,
            protocol_id=protocol.id,
            protocol_version=protocol.version,
            queries=queries,
            results_by_source=counts,
            duplicates_removed=removed,
        )
        return run, unique
