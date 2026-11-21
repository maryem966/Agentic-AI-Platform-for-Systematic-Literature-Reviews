"""Create ORM schema and immutable audit log."""

from __future__ import annotations

from pathlib import Path

from alembic import op
from sqlalchemy import inspect

from src.infrastructure.persistence.models import AuditLogORM

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

BASE_TABLES = (
    "research_projects",
    "research_protocols",
    "agent_executions",
    "human_reviews",
    "publications",
    "full_text_documents",
    "search_runs",
    "search_queries",
    "screening_decisions",
    "chunks",
    "extraction_records",
    "evidence",
    "report_versions",
    "claims",
    "claim_evidence_links",
)


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())
    found_base_tables = existing.intersection(BASE_TABLES)
    if not found_base_tables:
        sql_path = (
            Path(__file__).resolve().parents[5] / "docker" / "migrations" / "001_initial_schema.sql"
        )
        sql = sql_path.read_text(encoding="utf-8").replace("BEGIN;", "").replace("COMMIT;", "")
        for statement in sql.split(";"):
            if statement.strip():
                op.execute(statement.strip())
    elif not set(BASE_TABLES).issubset(existing):
        missing = sorted(set(BASE_TABLES) - existing)
        raise RuntimeError(f"Database has a partial SLR schema; missing tables: {missing}")

    # Docker's init SQL creates this too; checkfirst makes the Alembic bootstrap repeatable.
    AuditLogORM.__table__.create(bind=bind, checkfirst=True)
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_audit_log_mutation()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only; UPDATE and DELETE are forbidden';
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        DROP TRIGGER IF EXISTS trg_audit_log_immutable ON audit_log
    """)
    op.execute("""
        CREATE TRIGGER trg_audit_log_immutable
        BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE FUNCTION prevent_audit_log_mutation()
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_log CASCADE")
    for table in reversed(BASE_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    op.execute("DROP FUNCTION IF EXISTS prevent_audit_log_mutation()")
