"""SLR-Nexus Configuration."""
from .settings import settings, Settings
from .cypher_templates import CYPHER_TEMPLATES, ALLOWED_TEMPLATE_NAMES

__all__ = ["settings", "Settings", "CYPHER_TEMPLATES", "ALLOWED_TEMPLATE_NAMES"]
