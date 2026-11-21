"""OpenAlex Works search adapter."""

from __future__ import annotations

from typing import Any

import httpx

from ...config.settings import settings
from ...domain.entities.publication import Author, Publication
from .base import RetryingAPI, normalize_publication, valid_year


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        for offset in offsets:
            positions[offset] = word
    return " ".join(positions[position] for position in sorted(positions)) or None


class OpenAlexAdapter(RetryingAPI):
    def __init__(self, *, client: httpx.AsyncClient | None = None, **kwargs: Any) -> None:
        super().__init__(client=client, **kwargs)
        self.base_url = settings.openalex_base_url.rstrip("/")

    async def search(
        self,
        query: str,
        *,
        start_year: int | None = None,
        end_year: int | None = None,
        limit: int = 25,
    ) -> list[Publication]:
        if not query.strip() or not 1 <= limit <= 200:
            raise ValueError("query must be non-empty and limit must be between 1 and 200")
        params: dict[str, Any] = {"search": query, "per-page": limit}
        filters = []
        if start_year is not None:
            filters.append(f"from_publication_date:{start_year}-01-01")
        if end_year is not None:
            filters.append(f"to_publication_date:{end_year}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if settings.crossref_email:
            params["mailto"] = settings.crossref_email
        body = await self._get_json(f"{self.base_url}/works", params)
        publications = []
        for item in body.get("results", []):
            ids = item.get("ids") or {}
            openalex_id = (ids.get("openalex") or item.get("id") or "").rstrip("/").split("/")[-1]
            if not openalex_id:
                continue
            doi = item.get("doi")
            if doi:
                doi = doi.removeprefix("https://doi.org/").lower()
            authors = []
            for authorship in item.get("authorships", []):
                author = authorship.get("author") or {}
                if author.get("display_name"):
                    authors.append(Author(name=author["display_name"], orcid=author.get("orcid")))
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            publications.append(
                normalize_publication(
                    source="openalex",
                    title=item.get("title") or "Untitled",
                    external_ids={"openalex_id": openalex_id},
                    authors=authors,
                    year=valid_year(item.get("publication_year")),
                    abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
                    doi=doi,
                    url=location.get("landing_page_url") or item.get("id"),
                    journal=source.get("display_name"),
                )
            )
        return publications
