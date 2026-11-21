"""Adapters for scientific metadata search services."""

from .crossref_adapter import CrossrefAdapter
from .openalex_adapter import OpenAlexAdapter
from .semantic_scholar_adapter import SemanticScholarAdapter

__all__ = ["CrossrefAdapter", "OpenAlexAdapter", "SemanticScholarAdapter"]
