import operator
import weaviate
import json, re
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Any, Literal

from transformers import pipeline
import torch
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from concurrent.futures import ThreadPoolExecutor

from src.core.config import get_settings
from src.core.logger import get_logger
from src.core.llm_factory import get_llm
from src.services.retriever import retrieve
from src.utils.image_helpers import to_base64
from src.core.weaviate_client import get_weaviate_client
from src.agent.prompts import GUARDRAIL_PROMPT, ROUTER_PROMPT, REWRITER_PROMPT, GENERATOR_PROMPT

logger = get_logger(__name__)
_settings = get_settings()
# ── LLM Configuration 
# Each role gets its own explicit config. Fill in provider, model & params.
LLM_RAG_ARGS = {                # hard
    "provider": "groq",
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "temperature": 0.1,
    "top_p": 0.95,
}
LLM_REWRITER_ARGS = {           # medium
    "provider": "qwen/qwen3-32b",
    "model": "",
    "temperature": 0.3,
    "top_p": 0.9,
}
LLM_ROUTER_ARGS = {             # easy
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
    "temperature": 0,
}

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    guardrail_check: Literal["BENIGN", "MALICIOUS"]
    intention: Literal["GENERAL", "RAG"]
    collection_name: str  
    rewritten_queries: List[str]
    retrieved_documents: List[Any]

class AgenticRAG:
    def __init__(self):
        """
        Initialize the Agentic RAG core logic.
        Call setup() asynchronously to initialize the checkpointer and pool.
        """      
        self.llm_rag = get_llm(**LLM_RAG_ARGS)
        self.llm_rewriter = get_llm(**LLM_REWRITER_ARGS)
        self.llm_router = get_llm(**LLM_ROUTER_ARGS)

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
        # Initialize Prompt Guard Model efficiently
        self.guardrail_classifier = pipeline(
            "text-classification",
            model="meta-llama/Llama-Prompt-Guard-2-86M",
            token=settings.huggingface_api_token,
            device=0 if torch.cuda.is_available() else -1
        )
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

        # Safety guardrail (entry point — runs before anything else)
        builder.add_node("guardrail_check", self.guardrail_check)
        # Intention router
        builder.add_node("intent_router", self.intent_router)
        # RAG nodes
        builder.add_node('query_rewriter', self.query_rewriter)
        builder.add_node('retriever', self.retriever)
        builder.add_node('rag_generator', self.rag_generator)
        # General LLM
        builder.add_node("general_llm", self.general_llm)
        # Guardrail block (response node for unsafe inputs)
        builder.add_node("guardrail_block", self.guardrail_block)

        builder.set_entry_point('guardrail_check')
        # Guardrail gate: BENIGN → intent_router, MALICIOUS → guardrail_block
        builder.add_conditional_edges(
            "guardrail_check",
            lambda state: state.get('guardrail_check', 'BENIGN'),
            {
                "BENIGN": "intent_router",
                "MALICIOUS": "guardrail_block"
            }
        )
        # Intent router: RAG or GENERAL
        builder.add_conditional_edges(
            "intent_router",
            self.route,
            {
                "RAG": "query_rewriter",
                "GENERAL": "general_llm",
            }
        )
        builder.add_edge('query_rewriter', 'retriever')
        builder.add_edge('retriever', 'rag_generator')

        builder.add_edge('rag_generator', END)
        builder.add_edge('general_llm', END)
        builder.add_edge('guardrail_block', END)

        return builder
    
    def guardrail_check(self, state: AgentState):
        """Fast-fail safety gate. Runs before the intent router."""
        query = state.get('query', '')
        try:
            # Using model already loaded during setup()
            result = self.guardrail_classifier(query)
           
            decision = "MALICIOUS" if result[0]['label'] == "LABEL_1" else "BENIGN"
        except Exception as e:
            logger.error(f"Guardrail check failed: {e}. Defaulting to BENIGN.")
            decision = "BENIGN"
            
        return {'guardrail_check': decision}

    def intent_router(self, state: AgentState):
        """Classify safe queries as RAG or GENERAL."""
        query = state.get('query', '')
        history = state.get('messages', [])[-6:]

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

    def route(self, state: AgentState) -> str:
        """Helper function for the conditional edge to determine routing path."""
        return state.get('intention', 'RAG')

    def general_llm(self, state: AgentState):
            """Handle standard conversational inputs that require no documents."""
            query = state['query']
            history = state.get('messages', [])
            
            system_instruction = SystemMessage(
                content="You are a helpful AI assistant. Answer the user's conversational query naturally."
            )
            
            messages = [system_instruction] + history[-6:] + [HumanMessage(content=query)]
            response = self.llm_rag.invoke(messages)
            
            return {
                'messages': [HumanMessage(content=query), response]
            }

    def guardrail_block(self, state: AgentState):
        """Dead-end node that returns a safety refusal with the LLM's reason."""
        query = state['query']
        reason = state.get('violation_reason', 'violating safety guidelines')
        response = AIMessage(content=f"**I cannot process this request** — {reason}.")
        return {
            'messages': [HumanMessage(content=query), response]
        }

    def query_rewriter(self, state: AgentState):
        """Rewrite query to be more specific and context-aware"""
        query = state.get('query', '')
        history = state.get('messages', [])

        system_prompt = REWRITER_PROMPT

        # Build messages: system instructions → chat history → current query
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history[-6:])  # Last 3 turns of conversation context
        messages.append(HumanMessage(content=query))
        
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
            logger.warning(f"Query rewriting failed: {e}. Using original query.")
            queries = [query]
        
        logger.info(f"Rewritten queries: {queries}")
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

        logger.info(f"RAG response generated ({len(response.content)} chars)")
        
        return {
            'messages': [
                HumanMessage(content=query),
                response
            ]
        }

    def get_graph(self):
        """Hiển thị graph dưới dạng hình ảnh"""
        from IPython.display import Image, display
        
        img = self.graph.get_graph().draw_mermaid_png()
        return display(Image(img))