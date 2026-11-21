-- Append-only audit journal. Safe to run more than once.
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    research_id TEXT REFERENCES research_projects(id) ON DELETE SET NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action <> ''),
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_audit_resource
    ON audit_log(resource_type, resource_id, created_at);

CREATE OR REPLACE FUNCTION prevent_audit_log_mutation()
RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only; UPDATE and DELETE are forbidden';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_log_immutable ON audit_log;
CREATE TRIGGER trg_audit_log_immutable
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_log_mutation();
