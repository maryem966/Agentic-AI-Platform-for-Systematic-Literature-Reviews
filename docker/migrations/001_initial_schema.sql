-- SLR-Nexus initial PostgreSQL schema (aligned with src/domain/entities).
-- Apply once to an empty database. IDs remain TEXT to preserve external/source IDs.

BEGIN;

CREATE TABLE research_projects (
    id TEXT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    topic TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN
        ('draft','protocol_defined','searching','screening','extracting','writing','reviewing','completed','archived')),
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    protocol_id TEXT
);

CREATE TABLE research_protocols (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    research_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
    inclusion_criteria JSONB NOT NULL DEFAULT '[]'::jsonb,
    exclusion_criteria JSONB NOT NULL DEFAULT '[]'::jsonb,
    target_databases JSONB NOT NULL DEFAULT '["openalex","crossref","semantic_scholar"]'::jsonb,
    date_range JSONB,
    language_filter JSONB NOT NULL DEFAULT '[]'::jsonb,
    validated_by TEXT,
    validated_at TIMESTAMPTZ,
    modification_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    UNIQUE (research_id, version)
);
ALTER TABLE research_projects ADD CONSTRAINT fk_research_active_protocol
    FOREIGN KEY (protocol_id) REFERENCES research_protocols(id) ON DELETE SET NULL;

CREATE TABLE agent_executions (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    input_summary TEXT NOT NULL,
    output_summary TEXT,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','failed','escalated')),
    llm_calls INTEGER NOT NULL DEFAULT 0 CHECK (llm_calls >= 0),
    tokens_input INTEGER NOT NULL DEFAULT 0 CHECK (tokens_input >= 0),
    tokens_output INTEGER NOT NULL DEFAULT 0 CHECK (tokens_output >= 0),
    llm_provider TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE human_reviews (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    review_type TEXT NOT NULL CHECK (review_type IN
        ('protocol_validation','screening_arbitration','evidence_validation','neo4j_conflict','agent_disagreement','quality_escalation','report_approval')),
    reviewer_id TEXT NOT NULL,
    context_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    decision TEXT NOT NULL,
    justification TEXT NOT NULL,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    time_to_decision_seconds INTEGER CHECK (time_to_decision_seconds IS NULL OR time_to_decision_seconds >= 0)
);

CREATE TABLE publications (
    id TEXT PRIMARY KEY,
    external_ids JSONB NOT NULL DEFAULT '{}'::jsonb,
    title TEXT NOT NULL,
    authors JSONB NOT NULL DEFAULT '[]'::jsonb,
    year INTEGER,
    abstract TEXT,
    doi TEXT,
    url TEXT,
    source TEXT NOT NULL,
    journal TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX uq_publications_doi ON publications (lower(doi)) WHERE doi IS NOT NULL;

CREATE TABLE full_text_documents (
    id TEXT PRIMARY KEY,
    publication_id TEXT NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_hash CHAR(64) NOT NULL UNIQUE,
    extraction_status TEXT NOT NULL DEFAULT 'pending' CHECK (extraction_status IN ('pending','extracted','failed','inaccessible')),
    extraction_error TEXT,
    page_count INTEGER CHECK (page_count IS NULL OR page_count >= 0),
    extracted_at TIMESTAMPTZ,
    extractor_version TEXT
);
ALTER TABLE publications ADD COLUMN document_id TEXT REFERENCES full_text_documents(id) ON DELETE SET NULL;

CREATE TABLE search_runs (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    protocol_id TEXT NOT NULL REFERENCES research_protocols(id),
    protocol_version INTEGER NOT NULL CHECK (protocol_version > 0),
    executed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    results_by_source JSONB NOT NULL DEFAULT '{}'::jsonb,
    duplicates_removed INTEGER NOT NULL DEFAULT 0 CHECK (duplicates_removed >= 0),
    agent_execution_id TEXT REFERENCES agent_executions(id) ON DELETE SET NULL
);
CREATE TABLE search_queries (
    id TEXT PRIMARY KEY,
    search_run_id TEXT NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('openalex','crossref','semantic_scholar','manual')),
    query_string TEXT NOT NULL,
    filters_applied JSONB NOT NULL DEFAULT '{}'::jsonb,
    results_count INTEGER NOT NULL DEFAULT 0 CHECK (results_count >= 0),
    executed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    http_status INTEGER,
    error_message TEXT
);

CREATE TABLE screening_decisions (
    id TEXT PRIMARY KEY,
    publication_id TEXT NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    protocol_id TEXT NOT NULL REFERENCES research_protocols(id),
    screening_stage TEXT NOT NULL CHECK (screening_stage IN ('title_abstract','full_text')),
    decision TEXT NOT NULL CHECK (decision IN ('include','exclude','uncertain')),
    exclusion_reason TEXT,
    justification TEXT NOT NULL,
    criteria_applied JSONB NOT NULL DEFAULT '[]'::jsonb,
    agent TEXT NOT NULL CHECK (agent IN ('deterministic','llm')),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    human_review_id TEXT REFERENCES human_reviews(id) ON DELETE SET NULL,
    evaluator_2_decision_id TEXT REFERENCES screening_decisions(id) ON DELETE SET NULL,
    agreement_score DOUBLE PRECISION CHECK (agreement_score IS NULL OR agreement_score BETWEEN -1 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (decision <> 'exclude' OR exclusion_reason IS NOT NULL)
);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES full_text_documents(id) ON DELETE CASCADE,
    publication_id TEXT NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    page INTEGER CHECK (page IS NULL OR page > 0),
    section TEXT,
    section_type TEXT NOT NULL DEFAULT 'text' CHECK (section_type IN ('text','table','figure','formula','reference')),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    char_start INTEGER NOT NULL CHECK (char_start >= 0),
    char_end INTEGER NOT NULL CHECK (char_end >= char_start),
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE extraction_records (
    id TEXT PRIMARY KEY,
    publication_id TEXT NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    agent_execution_id TEXT NOT NULL REFERENCES agent_executions(id),
    extracted_fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    extraction_confidence DOUBLE PRECISION NOT NULL CHECK (extraction_confidence BETWEEN 0 AND 1),
    validated BOOLEAN NOT NULL DEFAULT FALSE,
    human_review_id TEXT REFERENCES human_reviews(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    publication_id TEXT NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    evidence_type TEXT NOT NULL CHECK (evidence_type IN ('result','method','limitation','dataset','metric','definition')),
    extracted_by_agent TEXT NOT NULL,
    extraction_confidence DOUBLE PRECISION NOT NULL CHECK (extraction_confidence BETWEEN 0 AND 1),
    verification_status TEXT NOT NULL DEFAULT 'extracted' CHECK (verification_status IN
        ('extracted','auto_verified','auto_rejected','human_validated','human_rejected','contested')),
    verification_method TEXT,
    human_review_id TEXT REFERENCES human_reviews(id) ON DELETE SET NULL,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE report_versions (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL REFERENCES research_projects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    content JSONB NOT NULL DEFAULT '{}'::jsonb,
    generated_by_agent TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    overall_quality_score DOUBLE PRECISION CHECK (overall_quality_score IS NULL OR overall_quality_score BETWEEN 0 AND 1),
    human_approved BOOLEAN NOT NULL DEFAULT FALSE,
    human_review_id TEXT REFERENCES human_reviews(id) ON DELETE SET NULL,
    previous_version_id TEXT REFERENCES report_versions(id) ON DELETE SET NULL,
    UNIQUE (research_id, version)
);

CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    rq_id TEXT NOT NULL,
    report_version_id TEXT NOT NULL REFERENCES report_versions(id) ON DELETE CASCADE,
    section TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('supported','partially_supported','contradicted','unsupported','pending')),
    verification_method TEXT NOT NULL DEFAULT 'semantic' CHECK (verification_method IN ('lexical','semantic','human','combined')),
    contradicting_evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE claim_evidence_links (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL CHECK (link_type IN ('supports','partially_supports','contradicts')),
    link_confidence DOUBLE PRECISION NOT NULL CHECK (link_confidence BETWEEN 0 AND 1),
    created_by TEXT NOT NULL CHECK (created_by IN ('auto','human')),
    UNIQUE (claim_id, evidence_id)
);

CREATE INDEX ix_protocols_research ON research_protocols(research_id);
CREATE INDEX ix_search_runs_research ON search_runs(research_id, executed_at DESC);
CREATE INDEX ix_screening_research_stage ON screening_decisions(research_id, screening_stage);
CREATE INDEX ix_chunks_publication ON chunks(publication_id);
CREATE INDEX ix_evidence_research_status ON evidence(research_id, verification_status);
CREATE INDEX ix_claims_report ON claims(report_version_id);

COMMIT;
