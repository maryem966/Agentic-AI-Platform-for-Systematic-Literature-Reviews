"""Deterministic token-window chunking with overlap and short-chunk merging."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..domain.entities.chunk import Chunk

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


class DocumentChunker:
    def __init__(self, size: int = 512, overlap: int = 64, minimum: int = 100) -> None:
        if size < 1 or overlap < 0 or overlap >= size or minimum < 1:
            raise ValueError("Invalid chunk size, overlap, or minimum")
        self.size, self.overlap, self.minimum = size, overlap, minimum

    def split(
        self,
        pages: list[PageText],
        *,
        document_id: str,
        publication_id: str,
        embedding_model: str,
        embedding_version: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        page_starts: list[tuple[int, int, PageText]] = []
        pieces: list[str] = []
        cursor = 0
        for page in pages:
            page_starts.append((cursor, cursor + len(page.text), page))
            pieces.append(page.text)
            cursor += len(page.text) + 1
        full_text = "\n".join(pieces)
        matches = list(TOKEN_RE.finditer(full_text))
        pending: tuple[PageText, int, int, str, list] | None = None
        step = self.size - self.overlap
        for start in range(0, len(matches), step):
            end_index = min(start + self.size, len(matches))
            if start >= end_index:
                break
            first, last = matches[start], matches[end_index - 1]
            text = full_text[first.start() : last.end()]
            page = next(
                (
                    candidate
                    for lower, upper, candidate in page_starts
                    if lower <= first.start() <= upper
                ),
                pages[-1],
            )
            item = (page, first.start(), last.end(), text, matches[start:end_index])
            if pending:
                prior, offset, _, prior_text, prior_tokens = pending
                if len(prior_tokens) < self.minimum:
                    merged = (
                        prior,
                        offset,
                        last.end(),
                        full_text[offset : last.end()],
                        prior_tokens + item[4],
                    )
                    pending = None
                    self._add(
                        chunks,
                        merged,
                        document_id,
                        publication_id,
                        embedding_model,
                        embedding_version,
                    )
                else:
                    self._add(
                        chunks,
                        pending,
                        document_id,
                        publication_id,
                        embedding_model,
                        embedding_version,
                    )
                    pending = None
                    self._add(
                        chunks,
                        item,
                        document_id,
                        publication_id,
                        embedding_model,
                        embedding_version,
                    )
            elif len(item[4]) < self.minimum:
                pending = item
            else:
                self._add(
                    chunks, item, document_id, publication_id, embedding_model, embedding_version
                )
            if end_index == len(matches):
                break
        if pending:
            if chunks:
                previous = chunks.pop()
                begin = previous.char_start
                self._add(
                    chunks,
                    (pending[0], begin, pending[2], full_text[begin : pending[2]], []),
                    document_id,
                    publication_id,
                    embedding_model,
                    embedding_version,
                )
            else:
                self._add(
                    chunks, pending, document_id, publication_id, embedding_model, embedding_version
                )
        return chunks

    @staticmethod
    def _section_type(text: str) -> Literal["text", "table", "formula", "figure", "reference"]:
        stripped = text.strip()
        if re.search(r"(?:\d+\s*\|\s*){2,}\d+", stripped) or stripped.count("\t") >= 2:
            return "table"
        if re.search(r"[=∑∫√±≤≥≠∞]|\\(?:frac|sum|int|sqrt)", stripped):
            return "formula"
        return "text"

    def _add(
        self,
        result: list[Chunk],
        item: tuple,
        document_id: str,
        publication_id: str,
        model: str,
        version: str,
    ) -> None:
        page, start, end, text, _ = item
        result.append(
            Chunk(
                document_id=document_id,
                publication_id=publication_id,
                text=text,
                page=page.page,
                section_type=self._section_type(text),
                chunk_index=len(result),
                char_start=start,
                char_end=end,
                embedding_model=model,
                embedding_version=version,
            )
        )
