from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel

from src.domain.ports.llm_port import LLMPort, LLMResponse
from src.infrastructure.llm.fallback import FallbackLLMAdapter
from src.infrastructure.llm.groq_adapter import GroqAdapter
from src.infrastructure.llm.ollama_adapter import OllamaAdapter
from src.infrastructure.scientific_apis.crossref_adapter import CrossrefAdapter
from src.infrastructure.scientific_apis.openalex_adapter import OpenAlexAdapter
from src.infrastructure.scientific_apis.semantic_scholar_adapter import SemanticScholarAdapter
from src.infrastructure.vector.qdrant_adapter import QdrantVectorStoreAdapter


@pytest.mark.asyncio
async def test_qdrant_upsert_uses_deterministic_id_and_restores_domain_id() -> None:
    client = SimpleNamespace(
        upsert=AsyncMock(),
        search=AsyncMock(return_value=[SimpleNamespace(
            id="deterministic-uuid", score=0.9,
            payload={"_slr_document_id": "chunk-1", "section": "Results"},
        )]),
        count=AsyncMock(return_value=SimpleNamespace(count=1)),
        delete=AsyncMock(),
        get_collections=AsyncMock(return_value=SimpleNamespace(collections=[])),
        create_collection=AsyncMock(),
        close=AsyncMock(),
    )
    store = QdrantVectorStoreAdapter(vector_size=3, client=client)
    await store.upsert("chunks", "chunk-1", [0.1, 0.2, 0.3], {"section": "Results"})
    point = client.upsert.await_args.kwargs["points"][0]
    assert point.id == store._point_id("chunks", "chunk-1")
    results = await store.search("chunks", [0.1, 0.2, 0.3], top_k=3)
    assert results[0].id == "chunk-1"
    assert results[0].payload == {"section": "Results"}
    with pytest.raises(ValueError, match="expects 3"):
        await store.search("chunks", [0.1, 0.2], top_k=3)


@pytest.mark.asyncio
async def test_qdrant_bootstraps_both_collections_idempotently() -> None:
    client = SimpleNamespace(
        get_collections=AsyncMock(return_value=SimpleNamespace(collections=[])),
        create_collection=AsyncMock(), close=AsyncMock(),
    )
    store = QdrantVectorStoreAdapter(vector_size=8, client=client)
    await store.ensure_collections()
    assert [call.kwargs["collection_name"] for call in client.create_collection.await_args_list] == [
        "chunks", "semantic_memory"
    ]


@pytest.mark.asyncio
async def test_ollama_adapter_maps_response_and_structured_json() -> None:
    client = SimpleNamespace(
        chat=AsyncMock(return_value=SimpleNamespace(
            model="test-model", message=SimpleNamespace(content='{"answer":"ok"}'),
            prompt_eval_count=4, eval_count=2,
        )),
        list=AsyncMock(return_value=object()),
    )
    adapter = OllamaAdapter(model="test-model", client=client)
    result = await adapter.invoke([{"role": "user", "content": "Hi"}])
    assert result == LLMResponse("{\"answer\":\"ok\"}", "test-model", "ollama", 4, 2)

    class Answer(BaseModel):
        answer: str

    parsed = await adapter.invoke_structured([{"role": "user", "content": "Answer"}], Answer)
    assert parsed.answer == "ok"


@pytest.mark.asyncio
async def test_groq_adapter_and_fallback_routing() -> None:
    completion = SimpleNamespace(
        model="groq-model", choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )
    create = AsyncMock(return_value=completion)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models=SimpleNamespace(list=AsyncMock(return_value=[])),
    )
    groq = GroqAdapter(model="groq-model", client=client)
    assert (await groq.invoke([{"role": "user", "content": "Hi"}])).tokens_input == 3

    primary = AsyncMock(spec=LLMPort)
    fallback = AsyncMock(spec=LLMPort)
    primary.provider_name = "ollama"
    primary.model_name = "local"
    fallback.provider_name = "groq"
    fallback.model_name = "remote"
    primary.invoke.side_effect = RuntimeError("offline")
    fallback.invoke.return_value = LLMResponse("ok", "remote", "groq")
    router = FallbackLLMAdapter(primary, fallback)
    response = await router.invoke([{"role": "user", "content": "Hi"}])
    assert response.provider == "groq"
    fallback.invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_scientific_adapters_normalize_mock_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "openalex" in str(request.url):
            return httpx.Response(200, json={"results": [{
                "id": "https://openalex.org/W123", "title": " A study ",
                "publication_year": 2020, "doi": "https://doi.org/10.1/test",
                "abstract_inverted_index": {"A": [0], "reproducible": [2], "study": [1]},
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "primary_location": {"landing_page_url": "https://example.org", "source": {"display_name": "Journal"}},
            }]})
        if "crossref" in str(request.url):
            return httpx.Response(200, json={"message": {"items": [{
                "DOI": "10.2/test", "title": ["Crossref title"], "author": [{"given": "Grace", "family": "Hopper"}],
                "published-print": {"date-parts": [[2022]]}, "URL": "https://doi.org/10.2/test",
            }]}})
        return httpx.Response(200, json={"data": [{
            "paperId": "S123", "title": "Semantic title", "year": 2021,
            "abstract": "An abstract", "authors": [{"name": "Katherine Johnson"}],
            "externalIds": {"DOI": "10.3/test"}, "url": "https://example.org",
        }]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapters = [OpenAlexAdapter(client=client), CrossrefAdapter(client=client), SemanticScholarAdapter(client=client)]
    results = [await adapter.search("test query", limit=2) for adapter in adapters]
    assert [items[0].source for items in results] == ["openalex", "crossref", "semantic_scholar"]
    assert [items[0].year for items in results] == [2020, 2022, 2021]
    assert results[0][0].doi == "10.1/test"
    assert results[0][0].abstract == "A study reproducible"
    await client.aclose()


@pytest.mark.asyncio
async def test_scientific_client_retries_rate_limit_with_retry_after() -> None:
    responses = iter([
        httpx.Response(429, headers={"Retry-After": "0"}, json={"message": "rate limited"}),
        httpx.Response(200, json={"results": []}),
    ])
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(responses)))
    adapter = OpenAlexAdapter(client=client, max_retries=1, backoff_seconds=0)
    assert await adapter.search("research") == []
    await client.aclose()
