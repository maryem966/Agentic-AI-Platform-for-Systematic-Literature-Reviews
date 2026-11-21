"""
Entités du domaine SLR-Nexus.
Aucune dépendance externe — uniquement Pydantic + stdlib.
"""

from .agent_execution import AgentExecution
from .chunk import Chunk
from .claim import Claim, ClaimEvidenceLink
from .evidence import Evidence
from .extraction_record import ExtractionRecord
from .full_text_document import FullTextDocument
from .human_review import HumanReview
from .publication import Author, Publication
from .report_version import ReportVersion
from .research_project import ResearchProject
from .research_protocol import (
    DateRange,
    ProtocolModification,
    ResearchProtocol,
    ResearchQuestion,
)
from .screening_decision import ScreeningDecision
from .search_run import SearchQuery, SearchRun

__all__ = [
    "ResearchProject",
    "ResearchQuestion",
    "DateRange",
    "ProtocolModification",
    "ResearchProtocol",
    "SearchQuery",
    "SearchRun",
    "Author",
    "Publication",
    "FullTextDocument",
    "ScreeningDecision",
    "Chunk",
    "ExtractionRecord",
    "Evidence",
    "Claim",
    "ClaimEvidenceLink",
    "ReportVersion",
    "AgentExecution",
    "HumanReview",
]
