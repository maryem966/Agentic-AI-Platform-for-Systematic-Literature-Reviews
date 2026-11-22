"""
Connecteurs d'APIs scientifiques réelles (OpenAlex & Crossref).
Inspiré et modernisé depuis real_tools.py de vos TD, avec conversion en entités Publication.
"""
from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from ...domain.entities.publication import Author, Publication


class ScientificAPIsClient:
    """Client pour interroger les APIs bibliographiques mondiales avec normalisation."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.headers = {"User-Agent": "SLR-Nexus/1.0 (mailto:research@slr-nexus.org)"}

    async def search_openalex(self, query: str, limit: int = 25) -> list[Publication]:
        """Recherche sur OpenAlex (catalogue de 250M d'articles)."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.openalex.org/works?search={encoded_query}&per-page={limit}"

        publications: list[Publication] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(url, headers=self.headers)
                if res.status_code != 200:
                    return []
                data = res.json()

            for item in data.get("results", []):
                doi = item.get("doi")
                if doi and doi.startswith("https://doi.org/"):
                    doi = doi.replace("https://doi.org/", "")

                authors = [
                    Author(name=authorship.get("author", {}).get("display_name", "Unknown"))
                    for authorship in item.get("authorships", [])
                ]

                # Reconstruction du résumé si disponible via abstract_inverted_index
                abstract = self._reconstruct_openalex_abstract(item.get("abstract_inverted_index"))

                pub = Publication(
                    external_ids={"openalex": item.get("id", "")},
                    title=item.get("title") or "Untitled",
                    authors=authors,
                    year=item.get("publication_year"),
                    abstract=abstract,
                    doi=doi,
                    url=item.get("doi") or item.get("id"),
                    source="openalex",
                    journal=item.get("primary_location", {}).get("source", {}).get("display_name"),
                )
                publications.append(pub)
        except Exception:
            return []

        return publications

    async def search_crossref(self, query: str, limit: int = 25) -> list[Publication]:
        """Recherche sur Crossref (registre mondial des DOI)."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.crossref.org/works?query={encoded_query}&rows={limit}"

        publications: list[Publication] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(url, headers=self.headers)
                if res.status_code != 200:
                    return []
                data = res.json()

            items = data.get("message", {}).get("items", [])
            for item in items:
                titles = item.get("title", [])
                title = titles[0] if titles else "Untitled"

                doi = item.get("DOI")
                authors = [
                    Author(name=f"{a.get('given', '')} {a.get('family', '')}".strip() or "Unknown")
                    for a in item.get("author", [])
                ]

                # Récupération de l'année
                published = item.get("published-print") or item.get("published-online") or {}
                date_parts = published.get("date-parts", [[]])[0]
                year = date_parts[0] if date_parts else None

                pub = Publication(
                    external_ids={"crossref": doi or ""},
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=item.get("abstract"),  # Peut contenir des balises XML
                    doi=doi,
                    url=item.get("URL"),
                    source="crossref",
                    journal=(item.get("container-title") or [None])[0],
                )
                publications.append(pub)
        except Exception:
            return []

        return publications

    def _reconstruct_openalex_abstract(self, inverted_index: dict[str, list[int]] | None) -> str | None:
        """OpenAlex stocke les résumés sous forme d'index inversé pour des raisons légales."""
        if not inverted_index:
            return None
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
