import operator
import weaviate

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.logging_config import get_logger
from src.retriever import retrieve
from src.utils import to_base64

logger = get_logger(__name__)

_settings = get_settings()

param_dict = {
    # aggragate documents, long context, deterministic, medium-large model
    'large_kwargs': { # grader, rag
        'model': 'gemini-2.5-flash-lite',
        'temperature': 0.3,
        'top_p': 0.5,
        'google_api_key': _settings.gemini_api_key
    },
    'small_kwargs': { # small, fast, deterministic, strict tokens
        'model': 'gemini-2.0-flash',
        'temperature': 0.3,
        'top_p': 0.5,
        'google_api_key': _settings.gemini_api_key
    },
}
  
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    collection_name: str  # Dynamic collection name passed per request
    queries: List[str]
    retrieved_documents: List[Any]


class AgenticRAG:
    def __init__(self):
        """
        Initialize the Agentic RAG system.
        Single instance that handles all collections via state.
        """      
        self.llm_rag = ChatGoogleGenerativeAI(**param_dict['large_kwargs'])
        self.llm_rewriter = ChatGoogleGenerativeAI(**param_dict['small_kwargs'])
        
        self.checkpoint = InMemorySaver()
        self.graph = self.build_graph()
        
    def build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)
        
        # RAG nodes
        builder.add_node('query_rewriter', self.query_rewriter)  
        builder.add_node('retriever', self.retriever)
        builder.add_node('generator', self.generator)
        
        builder.set_entry_point('query_rewriter')
        # Edges
        builder.add_edge('query_rewriter', 'retriever')
        builder.add_edge('retriever', 'generator')
        builder.add_edge('generator', END)
        
        return builder.compile(checkpointer=self.checkpoint)
    
    def query_rewriter(self, state: AgentState):
        """Rewrite query to be more specific and context-aware"""
        query = state.get('query', '')
        history = state.get('messages', [])

        system_prompt = """You are a Query Preprocessor. Your output must be a JSON array of strings, nothing else.

## Steps
1. **Resolve references**: Replace pronouns ("it", "that") with the actual topic from chat history. Skip if standalone.
2. **Clean**: Fix typos. Remove filler ("As a researcher...", "Could you please..."). Keep all entities, numbers, and terms exactly.
3. **Split only when** topics are completely unrelated and independently searchable. Otherwise keep as 1 query.

## Output format
Respond with ONLY a JSON array. No explanation. 1 to 3 items max.
Example: ["query one", "query two"]

## Examples

Input: "Why use LoRA?"
Output: ["Why use LoRA?"]

History: "Tell me about React hooks" / Input: "What about the useEffect one?"
Output: ["What about the useEffect hook?"]

Input: "As a data scientist, I'm curious about how T5-Large and BART-base compare on SQuAD 2.0 in F1 and exact match."
Output: ["How does T5-Large perform on SQuAD 2.0 in F1 and exact match?", "How does BART-base perform on SQuAD 2.0 in F1 and exact match?"]

Input: "How does BLIP handle image captioning, and what optimizer does ViT use for fine-tuning?"
Output: ["How does BLIP handle image captioning?", "What optimizer does ViT use for fine-tuning?"]

Input: "What are the accuracy scores for ResNet on CIFAR-10, CIFAR-100, and ImageNet?"
Output: ["What are the accuracy scores for ResNet on CIFAR-10, CIFAR-100, and ImageNet?"]
WHY: Same question across a list of items. Always 1 query, never split.

Input: "Compare GPT-4, Claude 3, Gemini, and Llama 3 on reasoning benchmarks"
Output: ["How do GPT-4 and Claude 3 perform on reasoning benchmarks?", "How do Gemini and Llama 3 perform on reasoning benchmarks?"]
WHY: 4 entities → merge into 2 pairs to stay ≤ 3.

REMEMBER: Output ONLY a JSON array. Maximum 3 queries. Same question across items = 1 query."""

        
        # Build messages: system instructions → chat history → current query
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history[-6:])  # Last 3 turns of conversation context
        messages.append(HumanMessage(content=query))
        
        import json, re
        
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
        
        return {'queries': queries}
    
    def retriever(self, state: AgentState):
        """Execute retrieval plan"""
        
        queries = state.get('queries', [])
        collection_name = state.get('collection_name', '')
        all_docs = []
        seen_ids = set()  # Track seen document IDs
        
        # Single connection shared across all sub-queries
        settings = get_settings()
        client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port
        )
        
        try:
            for query in queries:
                docs = retrieve(
                    query, 
                    collection_name=collection_name, 
                    top_k=25, 
                    top_k_reranker=7,
                    client=client
                )
                
                for doc in docs:
                    # Use UUID or chunk_id for deduplication
                    doc_id = str(doc.uuid)  # Weaviate objects have .uuid
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
        finally:
            client.close()
                
        if not all_docs:
            logger.warning(f"No documents retrieved for queries: {queries}")
        else:
            logger.info(f"Retrieved {len(all_docs)} unique documents")
        
        return {'retrieved_documents': all_docs}

    def _build_rag_messages(self, query: str, retrieved_documents: list) -> list:
        """Build the prompt messages for RAG generation (shared by generator and stream_generate)."""
        user_prompt = [
            {"type": "text", "text": "## Documents: \n\n"},
        ]

        # Format retrieved documents 
        for i, doc in enumerate(retrieved_documents, 1):
            props = doc.properties
            doc_type = props.get('type', '')
            source = props.get('source', '')
            page = props.get('page_number', '')
            image_path = props.get('image_path', '')
            source_ref = source + (f" (p.{page})" if page else "")
            
            if doc_type in ('Image', 'Table'):
                # Multimodal content - include caption and image
                caption = props.get('caption', '')
                text_part = f"[{i}] {caption}\nSource: {source_ref}\n\n"
                user_prompt.append({'type': 'text', 'text': text_part})

                base64_img = to_base64(image_path)
                if base64_img:
                    user_prompt.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{base64_img}'}
                    })
            else:
                text_part = f"[{i}] {props.get('text', '')}\nSource: {source_ref}\n\n"
                user_prompt.append({'type': 'text', 'text': text_part})
        
        user_prompt.append({'type': 'text', 'text': f"## Question:\n{query}"})

        rag_prompt = """# Role
You are an expert Research Assistant that answers questions by synthesizing information from provided documents. You produce well-structured, comprehensive responses with proper citations.

# Tasks
You have three primary tasks:

## Task 1: Synthesize Retrieved Documents
Analyze and combine information from multiple documents to answer the question:
- Answer based ONLY on the provided documents — do not use external knowledge
- Integrate information across sources to provide comprehensive, insightful answers
- If information is not found, explicitly state: "Not found in provided documents"
- Prioritize accuracy and completeness

## Task 2: Cite Sources
Provide inline citations to reference which documents support each claim:
- Use bracket notation: [1], [2], [3], etc.
- **Each citation must be in its own bracket** — write [1][2], NOT [1, 2]
- Place citations immediately after the relevant statement
- Cite every claim derived from the documents

## Task 3: Format Response as Markdown
Structure your response for readability:
- Use headers (##, ###) to organize sections when appropriate
- Use bullet points or numbered lists for multiple items
- Use **bold** for key terms and concepts
- Use code blocks for code snippets if applicable

# Examples

Example 1 - Proper citation format:
✓ Correct: "React hooks were introduced in version 16.8[1] and allow functional components to manage state[2]."
✗ Wrong: "React hooks were introduced in version 16.8 and allow functional components to manage state[1, 2]."

Example 2 - Multi-source synthesis:
"**OAuth 2.0** is the industry-standard protocol for authorization[1]. It works by issuing access tokens to third-party applications[2]. Common flows include Authorization Code for web apps[1] and Implicit for SPAs[3]."

Example 3 - Structured response:
"## Overview
The system uses a **microservices architecture**[1] with three main components:

### Components
- **API Gateway**: Handles routing and authentication[1][2]
- **User Service**: Manages user data and profiles[2]
- **Notification Service**: Sends emails and push notifications[3]"
"""
        
        return [
            SystemMessage(content=rag_prompt),
            HumanMessage(content=user_prompt)
        ]

    def generator(self, state: AgentState) -> None:
        """Generator aggregates retrieved documents (non-streaming)."""       
        query = state['query']
        retrieved_documents = state['retrieved_documents']

        messages = self._build_rag_messages(query, retrieved_documents)
        response = self.llm_rag.invoke(messages)

        logger.info(f"RAG response generated ({len(response.content)} chars)")
        
        return {
            'messages': [
                HumanMessage(content=query),  # Store user query too
                response                       # AI response
            ]
        }

    def get_graph(self):
        """Hiển thị graph dưới dạng hình ảnh"""
        from IPython.display import Image, display
        
        img = self.graph.get_graph().draw_mermaid_png()
        return display(Image(img))