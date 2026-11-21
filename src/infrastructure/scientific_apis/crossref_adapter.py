"""Crossref Works search adapter."""

from __future__ import annotations

from typing import Any

import httpx

from ...config.settings import settings
from ...domain.entities.publication import Author, Publication
from .base import RetryingAPI, normalize_publication, valid_year


class CrossrefAdapter(RetryingAPI):
    def __init__(self, *, client: httpx.AsyncClient | None = None, **kwargs: Any) -> None:
        super().__init__(client=client, **kwargs)
        self.base_url = settings.crossref_base_url.rstrip("/")

    async def search(
        self,
        query: str,
        *,
        start_year: int | None = None,
        end_year: int | None = None,
        limit: int = 25,
    ) -> list[Publication]:
        if not query.strip() or not 1 <= limit <= 1000:
            raise ValueError("query must be non-empty and limit must be between 1 and 1000")
        params: dict[str, Any] = {"query.bibliographic": query, "rows": limit}
        filters = []
        if start_year is not None:
            filters.append(f"from-pub-date:{start_year}-01-01")
        if end_year is not None:
            filters.append(f"until-pub-date:{end_year}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if settings.crossref_email:
            params["mailto"] = settings.crossref_email
        body = await self._get_json(f"{self.base_url}/works", params)
        items = (body.get("message") or {}).get("items", [])
        publications = []
        for item in items:
            doi = item.get("DOI")
            if not doi:
                continue
            published = item.get("published-print") or item.get("published-online") or {}
            parts = published.get("date-parts") or []
            year = valid_year(parts[0][0]) if parts and parts[0] else None
            authors = []
            for author in item.get("author", []):
                name = " ".join(filter(None, (author.get("given"), author.get("family"))))
                if name:
                    authors.append(Author(name=name, orcid=author.get("ORCID"), affiliation=None))
            titles = item.get("title") or []
            containers = item.get("container-title") or []
            publications.append(
                normalize_publication(
                    source="crossref",
                    title=titles[0] if titles else "Untitled",
                    external_ids={"doi": doi},
                    authors=authors,
                    year=year,
                    abstract=item.get("abstract"),
                    doi=doi.lower(),
                    url=item.get("URL"),
                    journal=containers[0] if containers else None,
                )
            )
        return publications
