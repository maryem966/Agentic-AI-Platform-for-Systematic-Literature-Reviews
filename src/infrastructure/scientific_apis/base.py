"""Shared HTTP retry policy and publication normalization helpers."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx

from ...config.settings import settings
from ...domain.entities.publication import Author, Publication


class RetryingAPI:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.max_retries = (
            settings.scientific_api_max_retries if max_retries is None else max_retries
        )
        self.backoff_seconds = (
            settings.scientific_api_backoff_seconds if backoff_seconds is None else backoff_seconds
        )
        if self.max_retries < 0 or self.backoff_seconds < 0:
            raise ValueError("retry count and backoff must be non-negative")
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.get(url, params=params)
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.backoff_seconds * (2**attempt))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self.max_retries:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = (
                        float(retry_after) if retry_after else self.backoff_seconds * (2**attempt)
                    )
                except ValueError:
                    delay = self.backoff_seconds * (2**attempt)
                await asyncio.sleep(max(0.0, delay))
                continue

            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Scientific API returned an unexpected JSON document")
            return data
        raise RuntimeError("retry loop exited unexpectedly")

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()


def stable_id(source: str, external_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slr-nexus:{source}:{external_id}"))


def normalize_publication(
    *,
    source: str,
    title: str,
    external_ids: dict[str, str],
    authors: list[Author],
    year: int | None = None,
    abstract: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    journal: str | None = None,
) -> Publication:
    source_id = next(iter(external_ids.values()), doi or title)
    return Publication(
        id=stable_id(source, source_id),
        external_ids=external_ids,
        title=title.strip(),
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        url=url,
        source=source,
        journal=journal,
    )


def valid_year(value: Any) -> int | None:
    return value if isinstance(value, int) and 1000 <= value <= 3000 else None
