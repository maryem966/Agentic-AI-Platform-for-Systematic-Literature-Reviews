"""Check that domain code does not import infrastructure/framework packages."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_IMPORTS: frozenset[str] = frozenset({
    "langchain", "langgraph", "fastapi", "sqlalchemy", "qdrant_client", "neo4j",
    "uvicorn", "streamlit", "alembic", "asyncpg", "psycopg2", "aiohttp", "httpx",
    "requests", "ollama", "groq", "pymupdf", "fitz", "sentence_transformers", "ragas",
    "structlog", "opentelemetry",
})


def check_file(path: Path) -> list[str]:
    """Return forbidden imports found in a Python source file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        return [f"ERROR {path}: SyntaxError - {error}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            if module.split(".")[0] in FORBIDDEN_IMPORTS:
                violations.append(f"ERROR {path}:{node.lineno} - forbidden import: {module}")
    return violations


def main() -> None:
    """Check every Python file under src/domain."""
    domain_path = Path(__file__).resolve().parent.parent / "src" / "domain"
    if not domain_path.exists():
        print(f"ERROR domain directory not found: {domain_path}")
        sys.exit(1)

    py_files = list(domain_path.rglob("*.py"))
    violations = [issue for path in py_files for issue in check_file(path)]
    if violations:
        print("Forbidden imports found in src/domain/")
        print("Note: pydantic is currently permitted for domain validation models.")
        print("\n".join(violations))
        sys.exit(1)
    print(f"OK: {len(py_files)} domain files contain no forbidden framework imports.")


if __name__ == "__main__":
    main()
