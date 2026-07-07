import asyncio
import json
import operator
import re
from functools import wraps
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.core.config import get_settings
from src.core.logger import get_logger
from src.models.base import get_llm
from src.components.reranker import get_reranker
from src.components.retriever import retrieve, resolve_reranker_mode
from src.components.image_utils import to_base64
from src.core.weaviate_client import get_weaviate_client
from src.prompts.prompts import (
    ROUTER_PROMPT,
    QUERY_RESOLVER_PROMPT,
    QUERY_DECOMPOSER_PROMPT,
    GENERATOR_PROMPT,
)

from src.core.telemetry import get_current_trace_id
from opentelemetry import trace

try:
    from openinference.semconv.trace import SpanAttributes

    SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
except ImportError:
    SPAN_KIND = "openinference.span.kind"

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

_RESOLVER_HISTORY_CHARS = 150
def trace_step(name, kind="CHAIN"):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(name) as span:
                    span.set_attribute(SPAN_KIND, kind)
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(name) as span:
                    span.set_attribute(SPAN_KIND, kind)
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    intention: Literal["CONVERSATIONAL", "INFORMATION_REQUEST"]
    collection_name: str
    resolved_query: str
    decomposed_queries: list[str]
    retrieved_documents: list[Any]


class AgenticRAG:
    def __init__(self):
        """
        Initialize the Agentic RAG core logic.
        Call setup() asynchronously to initialize the checkpointer and pool.
        """
        self.llm_rag = get_llm("rag_generator")
        self.llm_decomposer = get_llm("decomposer")
        self.llm_router = get_llm("router")
        self.llm_resolver = get_llm("resolver")

        self.pool = None
        self.checkpointer = None
        self.graph = None

        # Prepare the graph structure (uncompiled)
        self.builder = self.build_graph()

    async def setup(self):
        """Initialize Postgres connection and compile the graph."""
        settings = get_settings()

        logger.info(f"Connecting to Postgres for memory: {settings.pg_host}")
        self.pool = AsyncConnectionPool(
            conninfo=settings.pg_url,
            max_size=settings.pg_pool_max_size,
            kwargs={"autocommit": True},
            open=False,
        )
        await self.pool.open()
        self.checkpointer = AsyncPostgresSaver(self.pool)
        # Ensure tables exist
        await self.checkpointer.setup()
        # Compile the graph with persistent checkpointer
        self.graph = self.builder.compile(checkpointer=self.checkpointer)
        logger.info("AgenticRAG setup complete.")

    async def close(self):
        """Shutdown connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("AgenticRAG pool closed.")

    def build_graph(self) -> StateGraph:
        """Define the LangGraph state machine structure."""
        builder = StateGraph(AgentState)

        # Intention router
        builder.add_node("intent_router", self.intent_router)
        # RAG nodes
        builder.add_node("query_resolver", self.query_resolver)
        builder.add_node("query_decomposer", self.query_decomposer)
        builder.add_node("retriever", self.retriever)
        builder.add_node("rag_generator", self.rag_generator)
        # CONVERSATIONAL LLM
        builder.add_node("conversational_llm", self.conversational_llm)
        builder.set_entry_point("intent_router")
        # Intent router: INFORMATION_REQUEST or CONVERSATIONAL
        builder.add_conditional_edges(
            "intent_router",
            lambda state: state.get("intention", "INFORMATION_REQUEST"),
            {
                "INFORMATION_REQUEST": "query_resolver",
                "CONVERSATIONAL": "conversational_llm",
            },
        )
        builder.add_edge("query_resolver", "query_decomposer")
        builder.add_edge("query_decomposer", "retriever")
        builder.add_edge("retriever", "rag_generator")

        builder.add_edge("rag_generator", END)
        builder.add_edge("conversational_llm", END)

        return builder

    @trace_step("intent_router")
    async def intent_router(self, state: AgentState):
        """Classify safe queries as CONVERSATIONAL or INFORMATION_REQUEST."""
        query = state.get("query", "")

        # Only the last assistant message is needed â€” to detect "operate on previous response" intent
        last_assistant = next(
            (m for m in reversed(state.get("messages", [])) if not isinstance(m, HumanMessage)),
            None,
        )
        last_response_str = f'Last Assistant Response: "{last_assistant.content}"\n\n' if last_assistant else ""

        formatted_input = f'{last_response_str}Current Query: "{query}"'

        messages = [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=formatted_input)
        ]

        try:
            response = await self.llm_router.ainvoke(messages)
            intention = response.content.strip().upper()

            if intention not in ["INFORMATION_REQUEST", "CONVERSATIONAL"]:
                logger.warning(
                    f"Router output '{intention}' invalid. Defaulting to INFORMATION_REQUEST."
                )
                intention = "INFORMATION_REQUEST"
        except Exception as e:
            logger.error(f"Routing failed: {e}. Defaulting to INFORMATION_REQUEST.")
            intention = "INFORMATION_REQUEST"

        logger.info(f"Router classified intent as: {intention}")
        return {"intention": intention}

    @trace_step("conversational_llm")
    async def conversational_llm(self, state: AgentState):
        """Handle standard conversational inputs that require no documents."""
        query = state["query"]
        history = state.get("messages", [])

        system_instruction = SystemMessage(
            content="You are a helpful AI assistant. Respond in the same language as the user, Vietnamese or English. Be concise by default."
        )

        messages = [system_instruction] + history[-4:] + [HumanMessage(content=query)]
        response = await self.llm_rag.ainvoke(messages)

        return {"messages": [HumanMessage(content=query), response]}

    @trace_step("query_resolver")
    async def query_resolver(self, state: AgentState):
        """Resolve references using conversation history."""
        query = state.get("query", "")
        history = state.get("messages", [])

        if not history:
            logger.info(f"Resolved query: {query}")
            return {"resolved_query": query}

        history_str = ""
        for msg in history[-4:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            if role == "Assistant":
                content = content[:_RESOLVER_HISTORY_CHARS]
            history_str += f'{role}: "{content}"\n'
        formatted_input = f'History:\n{history_str.strip()}\n\nInput: "{query}"'

        messages = [
            SystemMessage(content=QUERY_RESOLVER_PROMPT),
            HumanMessage(content=formatted_input),
        ]

        try:
            raw_response = await self.llm_resolver.ainvoke(messages)
            resolved_query = raw_response.content.strip()
            if not resolved_query:
                logger.warning(
                    "LLM returned empty resolved query. Falling back to original query."
                )
                resolved_query = query
        except Exception as e:
            logger.warning(f"Query resolving failed: {e}. Using original query.")
            resolved_query = query

        logger.info(f"Resolved query: {resolved_query}")
        return {"resolved_query": resolved_query}

    @trace_step("query_decomposer")
    async def query_decomposer(self, state: AgentState):
        """Decompose query into sub-queries if necessary"""
        resolved_query = state.get("resolved_query", state.get("query", ""))

        messages = [
            SystemMessage(content=QUERY_DECOMPOSER_PROMPT),
            HumanMessage(content=resolved_query),
        ]

        try:
            raw_response = await self.llm_decomposer.ainvoke(messages)
            text = raw_response.content.strip()

            # Extract JSON array from response (handles extra text around it)
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                queries = [str(q).strip() for q in queries if str(q).strip()][:3]
                if not queries:
                    logger.warning(
                        "LLM returned empty queries list. Falling back to resolved query."
                    )
                    queries = [resolved_query]
            else:
                raise ValueError(f"No JSON array found in response: {text[:200]}")

        except Exception as e:
            logger.warning(f"Query decomposing failed: {e}. Using resolved query.")
            queries = [resolved_query]

        logger.info(f"Decomposed queries: {queries}")
        return {"decomposed_queries": queries}

    @trace_step("retriever", kind="RETRIEVER")
    async def retriever(self, state: AgentState):
        """Execute retrieval plan â€” sub-queries run concurrently.

        The Weaviate v4 sync client is thread-safe (gRPC channels support
        concurrent calls), so a single shared client is passed to each thread.
        Ref: https://weaviate.io/developers/weaviate/client-libraries/python/async
        """
        queries = state.get("decomposed_queries", [])
        collection_name = state.get("collection_name", "")
        settings = get_settings()
        reranker_mode = resolve_reranker_mode(settings.reranker_mode)
        final_top_k = settings.retrieval_top_k_reranker
        per_query_rerank_k = 0 if reranker_mode == "app" else final_top_k

        # Single connection shared across all concurrent sub-queries
        client = get_weaviate_client()

        async def _retrieve_one(query: str) -> list:
            return await asyncio.to_thread(
                retrieve,
                query,
                collection_name=collection_name,
                top_k=settings.retrieval_top_k,
                top_k_reranker=per_query_rerank_k,
                alpha=0.6,
                client=client,
            )

        try:
            results_per_query = await asyncio.gather(
                *[_retrieve_one(q) for q in queries]
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            results_per_query = []

        # Deduplicate across sub-queries, preserving order
        all_docs = []
        seen_ids: set = set()
        for docs in results_per_query:
            for doc in docs:
                doc_id = str(doc.uuid)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        if not all_docs:
            logger.warning(f"No documents retrieved for queries: {queries}")
        else:
            logger.info(f"Retrieved {len(all_docs)} unique documents")

        rerank_query = state.get("resolved_query", state.get("query", ""))
        final_docs = self._finalize_retrieved_docs(
            rerank_query,
            all_docs,
            reranker_mode,
            final_top_k,
            settings,
        )

        return {"retrieved_documents": final_docs}

    @staticmethod
    def _finalize_retrieved_docs(
        query: str,
        docs: list,
        reranker_mode: str,
        top_k: int,
        settings,
    ) -> list:
        if not docs or top_k <= 0:
            return docs
        if reranker_mode == "app":
            return get_reranker(settings).rerank(query, docs, top_k)
        return docs[:top_k]

    @staticmethod
    def _source_ref(source: str, page_number: int | None) -> str:
        if not page_number:
            return source
        return f"{source} (p.{page_number})"

    def _document_text_part(self, index: int, props: dict) -> str:
        doc_type = props.get("type", "").lower()
        label = {"image": "IMAGE", "table": "TABLE"}.get(doc_type, "TEXT")
        source_ref = self._source_ref(props.get("source", ""), props.get("page_number"))
        text = props.get("text") or ("no description available" if doc_type == "image" else "")
        return f"[{index}] [{label}]\nSource: {source_ref}\n{text}\n"

    @staticmethod
    def _text_only_content(content_blocks: list[dict]) -> str:
        return "".join(
            block["text"]
            for block in content_blocks
            if block.get("type") == "text"
        )

    def _build_rag_messages(self, query: str, retrieved_documents: list) -> list:
        """Build the prompt messages for RAG generation (shared by generator and stream_generate)."""
        has_images = False
        content_blocks = [
            {"type": "text", "text": "Documents: \n\n"},
        ]

        for i, doc in enumerate(retrieved_documents, 1):
            props = doc.properties
            text_part = self._document_text_part(i, props)
            content_blocks.append({"type": "text", "text": text_part})

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

        final_content = content_blocks if has_images else self._text_only_content(content_blocks)
        return [SystemMessage(content=GENERATOR_PROMPT), HumanMessage(content=final_content)]

    @trace_step("rag_generator")
    async def rag_generator(self, state: AgentState) -> dict:
        """Generator aggregates retrieved documents (streaming)."""
        query = state["query"]
        retrieved_documents = state["retrieved_documents"]

        messages = self._build_rag_messages(query, retrieved_documents)

        response = await self.llm_rag.ainvoke(messages)

        # Attach retrieved documents to the response message so it persists in the state
        formatted_docs = self._format_retrieved_docs(retrieved_documents)
        response.additional_kwargs["docs"] = formatted_docs

        logger.info(f"RAG response generated ({len(response.content)} chars)")

        return {"messages": [HumanMessage(content=query), response]}

    def _format_retrieved_docs(self, docs: list) -> list:
        """Extract and format retrieved documents from graph result."""
        retrieved_docs = []
        for doc in docs:
            props = doc.properties if hasattr(doc, "properties") else doc

            # Extract score from metadata if available
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

    async def chat(
        self, collection_name: str, message: str, session_id: str = "default"
    ) -> dict:
        """Execute normal chat workflow."""
        thread_id = f"{collection_name}:{session_id}"
        config = {"configurable": {"thread_id": thread_id}}

        result = await self.graph.ainvoke(
            {"query": message, "collection_name": collection_name}, config=config
        )

        response_text = ""
        if result.get("messages"):
            last_message = result["messages"][-1]
            response_text = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

        docs = result.get("retrieved_documents", [])
        return {
            "response": response_text,
            "retrieved_documents": self._format_retrieved_docs(docs),
        }

    async def stream_chat(
        self, collection_name: str, message: str, session_id: str = "default"
    ):
        """Stream the RAG response token-by-token using SSE."""
        tracer = trace.get_tracer(__name__)

        thread_id = f"{collection_name}:{session_id}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            with tracer.start_as_current_span("agent_chat"):
                trace_id = get_current_trace_id()

                async for event in self.graph.astream_events(
                    {
                        "query": message,
                        "collection_name": collection_name,
                    },
                    config=config,
                    version="v2",
                ):
                    sse = self._stream_event_to_sse(event)
                    if sse:
                        yield sse

                yield self._sse_event("done", trace_id=trace_id)

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield self._sse_event("error", message=str(e))

    @staticmethod
    def _sse_event(event_type: str, **payload) -> str:
        return f"data: {json.dumps({'type': event_type, **payload})}\n\n"

    def _stream_event_to_sse(self, event: dict) -> str | None:
        kind = event["event"]

        if kind == "on_chain_end" and event["name"] == "retriever":
            docs = event["data"]["output"].get("retrieved_documents", [])
            formatted_docs = self._format_retrieved_docs(docs)
            return self._sse_event("docs", documents=formatted_docs)

        if kind == "on_chat_model_stream":
            node_name = event.get("metadata", {}).get("langgraph_node")
            if node_name in ("rag_generator", "conversational_llm"):
                chunk_content = event["data"]["chunk"].content
                if chunk_content:
                    return self._sse_event("chunk", text=chunk_content)

        return None

    async def get_history(
        self, collection_name: str, session_id: str = "default"
    ) -> list:
        """Retrieve the conversational history from LangGraph PostgreSQL state."""
        thread_id = f"{collection_name}:{session_id}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state_snapshot = await self.graph.aget_state(config)

            # If no state exists yet
            if not state_snapshot or not state_snapshot.values:
                return []

            messages = state_snapshot.values.get("messages", [])
            formatted_history = []

            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)

                # Extract docs from AI message metadata if available
                docs = (
                    msg.additional_kwargs.get("docs", [])
                    if hasattr(msg, "additional_kwargs")
                    else []
                )

                formatted_history.append(
                    {"role": role, "content": content, "docs": docs}
                )

            return formatted_history
        except Exception as e:
            logger.error(f"Failed to fetch history for {thread_id}: {e}")
            return []

    async def clear_history(self, collection_name: str, session_id: str = "default"):
        """Clear the conversational history using LangGraph's checkpointer delete method."""
        thread_id = f"{collection_name}:{session_id}"
        logger.info(f"Clearing chat history for thread: {thread_id}")

        try:
            await self.checkpointer.adelete_thread(thread_id)
            return True
        except Exception as e:
            logger.error(f"Failed to clear history for {thread_id}: {e}")
            raise Exception(f"Failed to clear history: {e}")
