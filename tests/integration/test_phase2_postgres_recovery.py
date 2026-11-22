from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg import AsyncConnection

from src.orchestration.postgres_checkpointer import PostgreSQLCheckpointer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_POSTGRES_CHECKPOINTER_TESTS") != "1",
        reason="Set RUN_POSTGRES_CHECKPOINTER_TESTS=1 to use the local PostgreSQL service.",
    ),
]


class ApprovalState(TypedDict, total=False):
    requested: bool
    decision: str


def build_graph(checkpointer: object) -> object:
    def review(_state: ApprovalState) -> dict[str, str]:
        return {"decision": interrupt({"kind": "checkpoint_recovery"})}

    builder = StateGraph(ApprovalState)
    builder.add_node("review", review)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    return builder.compile(checkpointer=checkpointer)


def test_postgres_checkpoint_resumes_after_new_saver_instance() -> None:
    previous_policy = asyncio.get_event_loop_policy()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def run() -> None:
        from src.config.settings import settings

        thread_id = f"phase2-smoke-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        first = PostgreSQLCheckpointer()
        second: PostgreSQLCheckpointer | None = None
        checkpoint_schema_ready = False
        try:
            first_saver = await first.open()
            checkpoint_schema_ready = True
            graph = build_graph(first_saver)
            events = [event async for event in graph.astream({"requested": True}, config)]
            interrupt_event = next(
                event["__interrupt__"] for event in events if "__interrupt__" in event
            )
            assert interrupt_event[0].value["kind"] == "checkpoint_recovery"
            await first.close()

            # A fresh pool and saver model an application restart between approval and resume.
            second = PostgreSQLCheckpointer()
            second_saver = await second.open()
            resumed = await build_graph(second_saver).ainvoke(Command(resume="ACCEPT"), config)
            assert resumed["decision"] == "ACCEPT"
        finally:
            await first.close()
            if second is not None:
                await second.close()
            if checkpoint_schema_ready:
                conninfo = settings.resolved_database_url.replace(
                    "postgresql+asyncpg://", "postgresql://", 1
                )
                async with await AsyncConnection.connect(conninfo, autocommit=True) as connection:
                    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                        await connection.execute(
                            f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
                        )

    try:
        asyncio.run(run())
    finally:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(previous_policy)
