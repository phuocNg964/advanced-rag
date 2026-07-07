import asyncio
from types import SimpleNamespace

from src.agentic_rag import agent_workflow
from src.agentic_rag.agent_workflow import AgenticRAG
from src.components.reranker import Reranker


class FakeDoc:
    def __init__(self, doc_id, properties):
        self.uuid = doc_id
        self.properties = properties
        self.metadata = SimpleNamespace(score=None)


def _workflow_instance():
    return AgenticRAG.__new__(AgenticRAG)


def test_rag_message_uses_text_content_when_no_images(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent_workflow,
        "to_base64",
        lambda path: calls.append(path) or "encoded",
    )
    rag = _workflow_instance()
    docs = [
        FakeDoc(
            "1",
            {
                "type": "text",
                "text": "Text chunk",
                "image_path": "",
                "source": "paper.pdf",
            },
        )
    ]

    messages = rag._build_rag_messages("Summarize the method", docs)

    assert calls == []
    assert isinstance(messages[1].content, str)
    assert "Text chunk" in messages[1].content


def test_rag_message_attaches_all_retrieved_images(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent_workflow,
        "to_base64",
        lambda path: calls.append(path) or f"encoded-{path}",
    )
    rag = _workflow_instance()
    docs = [
        FakeDoc(
            "1",
            {"type": "image", "text": "First image", "image_path": "fig-1.png"},
        ),
        FakeDoc(
            "2",
            {"type": "text", "text": "Text chunk", "image_path": ""},
        ),
        FakeDoc(
            "3",
            {"type": "image", "text": "Second image", "image_path": "fig-2.png"},
        ),
        FakeDoc(
            "4",
            {"type": "image", "text": "Third image", "image_path": "fig-3.png"},
        ),
    ]

    messages = rag._build_rag_messages("Summarize the method", docs)
    image_blocks = [
        block for block in messages[1].content if block.get("type") == "image_url"
    ]

    assert calls == ["fig-1.png", "fig-2.png", "fig-3.png"]
    assert len(image_blocks) == 3


def test_app_rerank_happens_once_after_dedupe(monkeypatch):
    settings = SimpleNamespace(
        reranker_mode="app",
        retrieval_top_k=20,
        retrieval_top_k_reranker=5,
    )
    doc_a = FakeDoc("a", {"text": "A"})
    doc_b = FakeDoc("b", {"text": "B"})
    doc_c = FakeDoc("c", {"text": "C"})
    calls = []

    def fake_retrieve(query, **kwargs):
        calls.append((query, kwargs))
        if query == "q1":
            return [doc_a, doc_b]
        return [doc_b, doc_c]

    class FakeReranker:
        def __init__(self):
            self.calls = []

        def rerank(self, query, docs, top_k):
            self.calls.append((query, docs, top_k))
            return [doc_c, doc_a]

    reranker = FakeReranker()
    monkeypatch.setattr(agent_workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_workflow, "get_weaviate_client", lambda: object())
    monkeypatch.setattr(agent_workflow, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent_workflow, "get_reranker", lambda settings: reranker)

    rag = _workflow_instance()
    result = asyncio.run(
        rag.retriever(
            {
                "decomposed_queries": ["q1", "q2"],
                "collection_name": "Docs",
                "resolved_query": "global query",
            }
        )
    )

    assert [call[0] for call in calls] == ["q1", "q2"]
    assert all(call[1]["top_k"] == 20 for call in calls)
    assert all(call[1]["top_k_reranker"] == 0 for call in calls)
    assert reranker.calls == [("global query", [doc_a, doc_b, doc_c], 5)]
    assert result["retrieved_documents"] == [doc_c, doc_a]


def test_reranker_warmup_loads_model_once():
    calls = []
    settings = SimpleNamespace(
        app_reranker_model="reranker-model",
        app_reranker_device="cpu",
        app_reranker_batch_size=8,
    )

    class FakeModel:
        def predict(self, pairs, **kwargs):
            return [1.0 for _pair in pairs]

    def model_factory(model_name, device):
        calls.append((model_name, device))
        return FakeModel()

    reranker = Reranker(settings=settings, model_factory=model_factory)
    reranker.warmup()
    reranker.rerank("query", [FakeDoc("1", {"text": "text"})], 1)

    assert calls == [("reranker-model", "cpu")]


def test_agent_setup_uses_configured_postgres_pool_size(monkeypatch):
    captured = {}
    settings = SimpleNamespace(
        pg_host="postgres",
        pg_url="postgresql://postgres:postgres@postgres:5432/agentic-rag",
        pg_pool_max_size=5,
    )

    class FakePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def open(self):
            return None

    class FakeSaver:
        def __init__(self, pool):
            self.pool = pool

        async def setup(self):
            return None

    class FakeBuilder:
        def compile(self, checkpointer):
            return {"checkpointer": checkpointer}

    monkeypatch.setattr(agent_workflow, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_workflow, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(agent_workflow, "AsyncPostgresSaver", FakeSaver)

    rag = _workflow_instance()
    rag.pool = None
    rag.checkpointer = None
    rag.graph = None
    rag.builder = FakeBuilder()

    asyncio.run(rag.setup())

    assert captured["max_size"] == 5
