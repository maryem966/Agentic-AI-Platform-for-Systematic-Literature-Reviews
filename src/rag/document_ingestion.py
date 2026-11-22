"""Safe PDF retrieval and text extraction; source text remains byte-for-byte preserved."""

from __future__ import annotations

import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import httpx

from ..config.settings import settings


@dataclass(frozen=True)
class ExtractedPDF:
    data: bytes
    sha256: str
    pages: tuple[tuple[int, str], ...]
    prompt_injection_suspected: bool


class PDFIngestion:
    """Download public PDFs only, enforce size/type/signature, then extract with PyMuPDF."""

    def __init__(self, max_bytes: int | None = None, max_redirects: int = 5) -> None:
        self.max_bytes = max_bytes or settings.pdf_max_bytes
        self.max_redirects = max_redirects

    @staticmethod
    def _public_https(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("PDF URL must use public HTTPS without embedded credentials")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, 443)}
        except OSError as error:
            raise ValueError("PDF host could not be resolved") from error
        if not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise ValueError("PDF URL resolves to a non-public address")

    async def download(self, url: str) -> bytes:
        current = url
        timeout = httpx.Timeout(30, connect=10)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            for _ in range(self.max_redirects + 1):
                self._public_https(current)
                async with client.stream(
                    "GET", current, headers={"Accept": "application/pdf"}
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("PDF redirect has no location")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise ValueError(f"PDF server returned HTTP {response.status_code}")
                    mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if mime != "application/pdf":
                        raise ValueError(f"Expected application/pdf, received {mime or 'unknown'}")
                    length = response.headers.get("content-length")
                    if length and int(length) > self.max_bytes:
                        raise ValueError("PDF exceeds configured maximum size")
                    data = bytearray()
                    async for block in response.aiter_bytes():
                        data.extend(block)
                        if len(data) > self.max_bytes:
                            raise ValueError("PDF exceeds configured maximum size")
                    return bytes(data)
        raise ValueError("PDF redirect limit exceeded")

    async def fetch_doi(self, doi: str) -> bytes:
        return await self.download(f"https://doi.org/{quote(doi.strip(), safe='/')}")

    def extract(self, data: bytes) -> ExtractedPDF:
        if len(data) > self.max_bytes:
            raise ValueError("PDF exceeds configured maximum size")
        if not data.startswith(b"%PDF-"):
            raise ValueError("Downloaded content does not have a PDF signature")
        try:
            import fitz
        except ImportError as error:
            raise RuntimeError(
                "PyMuPDF is required for PDF extraction; install project requirements"
            ) from error
        try:
            with fitz.open(stream=data, filetype="pdf") as document:
                pages = tuple(
                    (number + 1, page.get_text("text")) for number, page in enumerate(document)
                )
                if document.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported")
        except (fitz.FileDataError, RuntimeError) as error:
            raise ValueError("PDF is corrupt, encrypted, or unreadable") from error
        text = "\n".join(content for _, content in pages)
        # This is a flag for later review; extracted source text is never modified or executed.
        phrases = (
            "ignore previous instructions",
            "system prompt",
            "reveal your prompt",
            "act as an assistant",
        )
        suspicious = any(phrase in text.casefold() for phrase in phrases)
        return ExtractedPDF(data, hashlib.sha256(data).hexdigest(), pages, suspicious)

    @staticmethod
    def persist(data: bytes, directory: Path, sha256: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{sha256}.pdf"
        if not target.exists():
            target.write_bytes(data)
        return target
