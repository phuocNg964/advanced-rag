import asyncio
import json
from functools import wraps
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from src.components.image_utils import to_base64
from src.components.reranker import get_reranker
from src.prompts.prompts import GENERATOR_PROMPT

try:
    from openinference.semconv.trace import SpanAttributes

    SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
except ImportError:
    SPAN_KIND = "openinference.span.kind"

tracer = trace.get_tracer(__name__)
_MULTI_QUERY_FINAL_TOP_K = 3
_AGENT_STEP_LABELS = {
    "intent_router": "Routing question",
    "query_resolver": "Resolving references",
    "query_decomposer": "Planning retrieval",
    "retriever": "Retrieving evidence",
    "rag_generator": "Generating cited answer",
    "conversational_llm": "Generating answer",
}


def trace_step(name, kind="CHAIN"):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(name) as span:
                    span.set_attribute(SPAN_KIND, kind)
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute(SPAN_KIND, kind)
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def rerank_k_for_retrieve(
    query_count: int,
    final_top_k: int,
    reranker_mode: str,
) -> int:
    if reranker_mode == "app":
        return 0
    return _per_query_top_k(query_count, final_top_k)


def unique_retrieved_docs(results_per_query: list[list[Any]]) -> list[Any]:
    return _dedupe_docs(_flatten_docs(results_per_query))


def message_content_to_text(content: Any) -> str:
    """Normalize LangChain string or content-block message payloads to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in {"thinking", "reasoning"}:
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def select_final_retrieved_docs(
    queries: list[str],
    results_per_query: list[list[Any]],
    single_query: str,
    reranker_mode: str,
    final_top_k: int,
    settings,
) -> list[Any]:
    if len(queries) <= 1:
        return _rerank_or_trim(
            single_query,
            unique_retrieved_docs(results_per_query),
            reranker_mode,
            final_top_k,
            settings,
        )

    per_query_top_k = _per_query_top_k(len(queries), final_top_k)
    if per_query_top_k <= 0:
        return unique_retrieved_docs(results_per_query)

    selected_docs = []
    reranker = get_reranker(settings) if reranker_mode == "app" else None
    for query, docs in zip(queries, results_per_query):
        selected_docs.extend(
            _rerank_or_trim(
                query,
                docs,
                reranker_mode,
                per_query_top_k,
                settings,
                reranker,
            )
        )

    return _dedupe_docs(selected_docs)


def build_rag_messages(query: str, retrieved_documents: list[Any]) -> list:
    """Build the prompt messages for RAG generation."""
    has_images = False
    content_blocks = [
        {"type": "text", "text": "Documents: \n\n"},
    ]

    for index, doc in enumerate(retrieved_documents, 1):
        props = doc.properties
        content_blocks.append(
            {"type": "text", "text": _document_text_part(index, props)}
        )

        doc_type = props.get("type", "").lower()
        if doc_type == "image":
            image_path = props.get("image_path", "")
            base64_img = to_base64(image_path)
            if base64_img:
                has_images = True
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                    }
                )

        content_blocks.append({"type": "text", "text": "---\n\n"})

    content_blocks.append({"type": "text", "text": f"Question:\n{query}"})

    final_content = content_blocks if has_images else _text_only_content(content_blocks)
    return [SystemMessage(content=GENERATOR_PROMPT), HumanMessage(content=final_content)]


def format_retrieved_docs(docs: list[Any]) -> list[dict]:
    """Extract and format retrieved documents from graph result."""
    retrieved_docs = []
    for doc in docs:
        props = doc.properties if hasattr(doc, "properties") else doc

        score = None
        if hasattr(doc, "metadata") and doc.metadata:
            score = getattr(doc.metadata, "score", None)

        retrieved_docs.append(
            {
                "text": props.get("text", ""),
                "source": props.get("source", ""),
                "page_number": props.get("page_number", 0),
                "section": props.get("section", ""),
                "type": props.get("type", ""),
                "image_path": props.get("image_path", ""),
                "score": score,
            }
        )
    return retrieved_docs


def sse_event(event_type: str, **payload) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


def stream_event_to_sse(event: dict) -> str | None:
    kind = event["event"]
    node_name = _event_node_name(event)

    if node_name in _AGENT_STEP_LABELS:
        if kind == "on_chain_start":
            return sse_event(
                "step",
                node=node_name,
                label=_AGENT_STEP_LABELS[node_name],
                status="running",
            )
        if kind == "on_chain_end":
            step_sse = sse_event(
                "step",
                node=node_name,
                label=_AGENT_STEP_LABELS[node_name],
                status="completed",
            )
            if node_name == "retriever":
                docs = event["data"]["output"].get("retrieved_documents", [])
                formatted_docs = format_retrieved_docs(docs)
                return step_sse + sse_event("docs", documents=formatted_docs)
            return step_sse
        if kind == "on_chain_error":
            return sse_event(
                "step",
                node=node_name,
                label=_AGENT_STEP_LABELS[node_name],
                status="failed",
            )

    if kind == "on_chat_model_stream":
        if node_name in ("rag_generator", "conversational_llm"):
            chunk_content = message_content_to_text(event["data"]["chunk"].content)
            if chunk_content:
                return sse_event("chunk", text=chunk_content)

    return None


def _event_node_name(event: dict) -> str:
    metadata_node = event.get("metadata", {}).get("langgraph_node")
    if metadata_node:
        return metadata_node
    return event.get("name", "")


def _per_query_top_k(query_count: int, final_top_k: int) -> int:
    if query_count > 1:
        return min(_MULTI_QUERY_FINAL_TOP_K, final_top_k)
    return final_top_k


def _flatten_docs(results_per_query: list[list[Any]]) -> list[Any]:
    return [doc for docs in results_per_query for doc in docs]


def _dedupe_docs(docs: list[Any]) -> list[Any]:
    deduped = []
    seen_ids: set[str] = set()
    for doc in docs:
        doc_id = str(doc.uuid)
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        deduped.append(doc)
    return deduped


def _rerank_or_trim(
    query: str,
    docs: list[Any],
    reranker_mode: str,
    top_k: int,
    settings,
    reranker=None,
) -> list[Any]:
    if not docs or top_k <= 0:
        return docs
    if reranker_mode == "app":
        reranker = reranker or get_reranker(settings)
        return reranker.rerank(query, docs, top_k)
    return docs[:top_k]


def _source_ref(source: str, page_number: int | None) -> str:
    if not page_number:
        return source
    return f"{source} (p.{page_number})"


def _document_text_part(index: int, props: dict) -> str:
    doc_type = props.get("type", "").lower()
    label = {"image": "IMAGE", "table": "TABLE"}.get(doc_type, "TEXT")
    source_ref = _source_ref(props.get("source", ""), props.get("page_number"))
    text = props.get("text") or (
        "no description available" if doc_type == "image" else ""
    )
    return f"[{index}] [{label}]\nSource: {source_ref}\n{text}\n"


def _text_only_content(content_blocks: list[dict]) -> str:
    return "".join(
        block["text"]
        for block in content_blocks
        if block.get("type") == "text"
    )
