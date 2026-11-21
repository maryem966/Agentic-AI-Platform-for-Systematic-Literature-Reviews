"""Value Objects : ScreeningStage, ScreeningDecisionValue"""

from enum import Enum


class ScreeningStage(str, Enum):
    """Étape du screening dans la SLR."""

    TITLE_ABSTRACT = "title_abstract"
    FULL_TEXT = "full_text"


class ScreeningDecisionValue(str, Enum):
    """Décision de screening possible."""

    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"
