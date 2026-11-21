"""
Entités du domaine SLR-Nexus.
Aucune dépendance externe — uniquement Pydantic + stdlib.
"""
from .research_project import ResearchProject
from .research_protocol import (
    DateRange,
    ProtocolModification,
    ResearchProtocol,
    ResearchQuestion,
)
from .search_run import SearchQuery, SearchRun
from .publication import Author, Publication
from .full_text_document import FullTextDocument
from .screening_decision import ScreeningDecision
from .chunk import Chunk
from .extraction_record import ExtractionRecord
from .evidence import Evidence
from .claim import Claim, ClaimEvidenceLink
from .report_version import ReportVersion
from .agent_execution import AgentExecution
from .human_review import HumanReview

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
