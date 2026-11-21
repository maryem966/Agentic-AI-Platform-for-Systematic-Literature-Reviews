"""Semantic Scholar Graph API search adapter."""

from __future__ import annotations

from typing import Any

import httpx

from ...config.settings import settings
from ...domain.entities.publication import Author, Publication
from .base import RetryingAPI, normalize_publication, valid_year


class SemanticScholarAdapter(RetryingAPI):
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(client=client, **kwargs)
        self.base_url = settings.semantic_scholar_base_url.rstrip("/")
        key = api_key if api_key is not None else settings.semantic_scholar_api_key
        if key:
            self.client.headers.update({"x-api-key": key})

    async def search(
        self,
        query: str,
        *,
        start_year: int | None = None,
        end_year: int | None = None,
        limit: int = 25,
    ) -> list[Publication]:
        if not query.strip() or not 1 <= limit <= 100:
            raise ValueError("query must be non-empty and limit must be between 1 and 100")
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,year,authors,externalIds,url,venue",
        }
        if start_year and end_year:
            params["year"] = f"{start_year}-{end_year}"
        elif start_year:
            params["year"] = f"{start_year}-"
        elif end_year:
            params["year"] = f"-{end_year}"
        body = await self._get_json(f"{self.base_url}/paper/search", params)
        publications = []
        for item in body.get("data", []):
            paper_id = item.get("paperId")
            if not paper_id:
                continue
            ext = item.get("externalIds") or {}
            identifiers = {"semantic_scholar_id": paper_id}
            if ext.get("DOI"):
                identifiers["doi"] = ext["DOI"]
            authors = [
                Author(name=author["name"])
                for author in item.get("authors", [])
                if author.get("name")
            ]
            doi = ext.get("DOI")
            publications.append(
                normalize_publication(
                    source="semantic_scholar",
                    title=item.get("title") or "Untitled",
                    external_ids=identifiers,
                    authors=authors,
                    year=valid_year(item.get("year")),
                    abstract=item.get("abstract"),
                    doi=doi.lower() if doi else None,
                    url=item.get("url"),
                    journal=item.get("venue"),
                )
            )
        return publications
