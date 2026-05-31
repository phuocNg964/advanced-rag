import operator
import weaviate
import json, re
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Any, Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from concurrent.futures import ThreadPoolExecutor

from src.core.config import get_settings
from src.core.logger import get_logger
from src.models.base import get_llm
from src.components.retriever import retrieve
from src.components.parser import to_base64
from src.core.weaviate_client import get_weaviate_client
from src.prompts.prompts import ROUTER_PROMPT, QUERY_RESOLVER_PROMPT, QUERY_DECOMPOSER_PROMPT, GENERATOR_PROMPT

from src.core.telemetry import get_current_trace_id
from opentelemetry import trace

logger = get_logger(__name__)
_settings = get_settings()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    intention: Literal["GENERAL", "RAG"]
    collection_name: str  
    resolved_query: str
    rewritten_queries: List[str]
    retrieved_documents: List[Any]

class AgenticRAG:
    def __init__(self):
        """
        Initialize the Agentic RAG core logic.
        Call setup() asynchronously to initialize the checkpointer and pool.
        """      
        self.llm_rag = get_llm("rag_generator")
        self.llm_rewriter = get_llm("rewriter")
        self.llm_router = get_llm("router")

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
            max_size=20,
            kwargs={"autocommit": True},
            open=False
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
        builder.add_node('query_resolver', self.query_resolver)
        builder.add_node('query_decomposer', self.query_decomposer)
        builder.add_node('retriever', self.retriever)
        builder.add_node('rag_generator', self.rag_generator)
        # General LLM
        builder.add_node("general_llm", self.general_llm)
        builder.set_entry_point('intent_router')
        # Intent router: RAG or GENERAL
        builder.add_conditional_edges(
            "intent_router",
            lambda state: state.get("intention", "RAG"),
            {
                "RAG": "query_resolver",
                "GENERAL": "general_llm",
            }
        )
        builder.add_edge('query_resolver', 'query_decomposer')
        builder.add_edge('query_decomposer', 'retriever')
        builder.add_edge('retriever', 'rag_generator')

        builder.add_edge('rag_generator', END)
        builder.add_edge('general_llm', END)

        return builder

    def intent_router(self, state: AgentState):
        """Classify safe queries as RAG or GENERAL."""
        query = state.get('query', '')
        history = state.get('messages', [])[-4:]

        messages = [SystemMessage(content=ROUTER_PROMPT)]
        messages.extend(history)
        messages.append(HumanMessage(content=query))

        try:
            response = self.llm_router.invoke(messages)
            intention = response.content.strip().upper()

            if intention not in ["GENERAL", "RAG"]:
                logger.warning(f"Router output '{intention}' invalid. Defaulting to RAG.")
                intention = "RAG"
        except Exception as e:
            logger.error(f"Routing failed: {e}. Defaulting to RAG.")
            intention = "RAG"

        logger.info(f"Router classified intent as: {intention}")
        return {'intention': intention}

    def general_llm(self, state: AgentState):
        """Handle standard conversational inputs that require no documents."""
        query = state['query']
        history = state.get('messages', [])
        
        system_instruction = SystemMessage(
            content="You are a helpful AI assistant. Answer the user's conversational query naturally."
        )
        
        messages = [system_instruction] + history[-4:] + [HumanMessage(content=query)]
        response = self.llm_rag.invoke(messages)
        
        return {
            'messages': [HumanMessage(content=query), response]
        }

    def query_resolver(self, state: AgentState):
        """Resolve references and remove filler from query"""
        query = state.get('query', '')
        history = state.get('messages', [])

        system_prompt = QUERY_RESOLVER_PROMPT

        # Format history into a string to match the prompt's expected format
        history_str = ""
        for msg in history[-4:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            history_str += f"{role}: \"{content}\"\n"
            
        formatted_input = f"History:\n{history_str.strip()}\n\nInput: \"{query}\""

        # Build messages: system instructions → formatted text input
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_input)
        ]
        
        try:
            raw_response = self.llm_rewriter.invoke(messages)
            resolved_query = raw_response.content.strip()
        except Exception as e:
            logger.warning(f"Query resolving failed: {e}. Using original query.")
            resolved_query = query
            
        logger.info(f"Resolved query: {resolved_query}")
        return {'resolved_query': resolved_query}

    def query_decomposer(self, state: AgentState):
        """Decompose query into sub-queries if necessary"""
        resolved_query = state.get('resolved_query', state.get('query', ''))

        system_prompt = QUERY_DECOMPOSER_PROMPT

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=resolved_query)
        ]
        
        try:
            raw_response = self.llm_rewriter.invoke(messages)
            text = raw_response.content.strip()
            
            # Extract JSON array from response (handles extra text around it)
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                queries = [str(q) for q in queries if q][:3]
            else:
                raise ValueError(f"No JSON array found in response: {text[:200]}")
            
        except Exception as e:
            logger.warning(f"Query decomposing failed: {e}. Using resolved query.")
            queries = [resolved_query]
        
        logger.info(f"Decomposed queries: {queries}")
        return {'rewritten_queries': queries}
    
    def retriever(self, state: AgentState):
        """Execute retrieval plan"""
        
        queries = state.get('rewritten_queries', [])
        collection_name = state.get('collection_name', '')
        all_docs = []
        seen_ids = set()  # Track seen document IDs
        
        # Single connection shared across all sub-queries
        client = get_weaviate_client()
        
        try:            
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Issue all sub-queries to Weaviate in parallel
                futures = [
                    executor.submit(
                        retrieve,
                        query,
                        collection_name=collection_name,
                        top_k=25,
                        top_k_reranker=7,
                        client=client
                    ) for query in queries
                ]
                
                for future in futures:
                    docs = future.result()
                    for doc in docs:
                        # Use UUID or chunk_id for deduplication
                        doc_id = str(doc.uuid)
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_docs.append(doc)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
                
        if not all_docs:
            logger.warning(f"No documents retrieved for queries: {queries}")
        else:
            logger.info(f"Retrieved {len(all_docs)} unique documents")
        
        return {'retrieved_documents': all_docs}

    def _build_rag_messages(self, query: str, retrieved_documents: list) -> list:
        """Build the prompt messages for RAG generation (shared by generator and stream_generate)."""
        user_prompt = [
            {"type": "text", "text": "Documents: \n\n"},
        ]

        # Format retrieved documents 
        for i, doc in enumerate(retrieved_documents, 1):
            props = doc.properties
            doc_type = props.get('type', '')
            source = props.get('source', '')
            page = props.get('page_number', '')
            source_ref = source + (f" (p.{page})" if page else "")
            
            # Configure image/table chunks
            if doc_type in ('Image', 'Table'):
                image_path = props.get('image_path', '')
                caption = props.get('caption', 'no description available')
                text_part = f"[{i}] [{'IMAGE' if doc_type == 'Image' else 'TABLE'}] {caption}\n(Source: {source_ref})\n"
                user_prompt.append({'type': 'text', 'text': text_part})

                base64_img = to_base64(image_path)
                if base64_img:
                    user_prompt.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{base64_img}'}
                    })
            # Configure text chunks
            else:
                text_part = f"[{i}] {props.get('text', '')}\n(Source: {source_ref})\n"
                user_prompt.append({'type': 'text', 'text': text_part})
            
            user_prompt.append({'type': 'text', 'text': '---\n'})
        
        user_prompt.append({'type': 'text', 'text': f"Question:\n{query}"})

        rag_prompt = GENERATOR_PROMPT
        
        return [
            SystemMessage(content=rag_prompt),
            HumanMessage(content=user_prompt)
        ]

    async def rag_generator(self, state: AgentState) -> dict:
        """Generator aggregates retrieved documents (streaming)."""       
        query = state['query']
        retrieved_documents = state['retrieved_documents']

        messages = self._build_rag_messages(query, retrieved_documents)
        
        response = await self.llm_rag.ainvoke(messages)
        
        # Attach retrieved documents to the response message so it persists in the state
        formatted_docs = self._format_retrieved_docs(retrieved_documents)
        response.additional_kwargs["docs"] = formatted_docs

        logger.info(f"RAG response generated ({len(response.content)} chars)")
        
        return {
            'messages': [
                HumanMessage(content=query),
                response
            ]
        }

    def _format_retrieved_docs(self, docs: list) -> list:
        """Extract and format retrieved documents from graph result."""
        retrieved_docs = []
        for doc in docs:
            props = doc.properties if hasattr(doc, 'properties') else doc
            
            # Extract score from metadata if available
            score = None
            if hasattr(doc, 'metadata') and doc.metadata:
                score = getattr(doc.metadata, 'score', None)
                
            retrieved_docs.append({
                "text": props.get("text", ""),
                "source": props.get("source", ""),
                "page_number": props.get("page_number", ""),
                "type": props.get("type", ""),
                "image_path": props.get("image_path", ""),
                "score": score
            })
        return retrieved_docs

    async def chat(self, collection_name: str, message: str, session_id: str = "default") -> dict:
        """Execute normal chat workflow."""
        thread_id = f"{collection_name}:{session_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        result = await self.graph.ainvoke(
            {"query": message, "collection_name": collection_name},
            config=config
        )
        
        response_text = ""
        if result.get("messages"):
            last_message = result["messages"][-1]
            response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
        docs = result.get("retrieved_documents", [])
        return {
            "response": response_text,
            "retrieved_documents": self._format_retrieved_docs(docs)
        }

    async def stream_chat(self, collection_name: str, message: str, session_id: str = "default"):
        """Stream the RAG response token-by-token using SSE."""
        tracer = trace.get_tracer(__name__)
        
        thread_id = f"{collection_name}:{session_id}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            with tracer.start_as_current_span("agent_chat") as span:
                trace_id = get_current_trace_id()
                
                async for event in self.graph.astream_events(
                    {
                        "query": message,
                        "collection_name": collection_name,
                    },
                    config=config,
                    version='v2'
                ):
                    kind = event['event']

                    if kind == "on_chain_end" and event["name"] == "retriever":
                        docs = event["data"]['output'].get("retrieved_documents", [])
                        formatted_docs = self._format_retrieved_docs(docs)
                        yield f"data: {json.dumps({'type': 'docs', 'documents': formatted_docs})}\n\n"

                    elif kind == "on_chat_model_stream":
                        node_name = event.get("metadata", {}).get("langgraph_node")
                        if node_name in ("rag_generator", "general_llm"):
                            chunk_content = event['data']['chunk'].content
                            if chunk_content:
                                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_content})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'trace_id': trace_id})}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    def get_graph(self):
        """Hiển thị graph dưới dạng hình ảnh"""
        from IPython.display import Image, display

        img = self.graph.get_graph().draw_mermaid_png()
        return display(Image(img))

    async def get_history(self, collection_name: str, session_id: str = "default") -> list:
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
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # Extract docs from AI message metadata if available
                docs = msg.additional_kwargs.get("docs", []) if hasattr(msg, 'additional_kwargs') else []
                
                formatted_history.append({
                    "role": role,
                    "content": content,
                    "docs": docs
                })
                
            return formatted_history
        except Exception as e:
            logger.error(f"Failed to fetch history for {thread_id}: {e}")
            return []

    async def clear_history(self, collection_name: str, session_id: str = "default"):
        """Clear the conversational history using LangGraph's checkpointer delete method."""
        thread_id = f"{collection_name}:{session_id}"
        logger.info(f"Clearing chat history for thread: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            await self.checkpointer.adelete_thread(config)
            return True
        except Exception as e:
            logger.error(f"Failed to clear history for {thread_id}: {e}")
            raise Exception(f"Failed to clear history: {e}")
