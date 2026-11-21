"""SLR-Nexus Configuration."""

from .cypher_templates import ALLOWED_TEMPLATE_NAMES, CYPHER_TEMPLATES
from .settings import Settings, settings

__all__ = ["settings", "Settings", "CYPHER_TEMPLATES", "ALLOWED_TEMPLATE_NAMES"]
