# Phase 2 — Orchestration

## Current workflow

`POST /api/v1/research/{research_id}/run` reads an existing research project, creates a
structured protocol with the configured LLM, saves it, and pauses at `protocol_validation`.
`POST /api/v1/research/{research_id}/validate` resumes the same LangGraph thread with
`ACCEPT`, `REVISE`, or `ESCALATE`. `ACCEPT` validates the protocol and launches the three
scientific API adapters. Search queries, filters, counts, errors, the `SearchRun`, and
deduplicated publications are persisted. If no publications are found, the workflow pauses
again and permits `REVISE` or `ESCALATE`.

The protocol review happens before search so that an unapproved protocol cannot trigger
external API calls. It is the human approval point described by the phase tasks.

## PostgreSQL checkpoints

The workflow uses LangGraph's official `AsyncPostgresSaver`; its `setup()` method creates
the saver-managed checkpoint tables. Those tables are separate from SLR business tables,
and the research ID is used as the checkpoint `thread_id` so the API can resume the same run.
The psycopg pool uses autocommit and dictionary rows as required by the saver. On Windows,
the application selects `WindowsSelectorEventLoopPolicy` because psycopg's async pool does
not work with the default Proactor loop.

## Starting locally

Install the versions in `requirements.txt`, configure PostgreSQL and at least one LLM provider
in `.env`, start PostgreSQL, and launch the API:

```powershell
uvicorn src.api.main:app --reload
```

Create a research project first through the Phase 1 use case/repository. Then call the `run`
endpoint and submit the decision returned by the pending interrupt to the `validate` endpoint.
The later screening, indexing, extraction, writing, critic, and quality nodes are intentionally
stubs in this phase.

To verify recovery against the real PostgreSQL service, configure `.env` to match the initialized
database role and run:

```powershell
$env:RUN_POSTGRES_CHECKPOINTER_TESTS = "1"
python -m pytest tests/integration/test_phase2_postgres_recovery.py -q
```
