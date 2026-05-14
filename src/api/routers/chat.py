import json
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.core.telemetry import get_current_trace_id

logger = get_logger(__name__)

router = APIRouter()

# Will be initialized by main.py
rag = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, max_length=100)

class ChatResponse(BaseModel):
    response: str
    retrieved_documents: List[Dict[str, Any]]

def _format_retrieved_docs(result: dict) -> List[Dict[str, Any]]:
    """Extract and format retrieved documents from graph result."""
    retrieved_docs = []
    for doc in result.get("retrieved_documents", []):
        props = doc.properties if hasattr(doc, 'properties') else doc
        retrieved_docs.append({
            "text": props.get("text", ""),
            "source": props.get("source", ""),
            "page_number": props.get("page_number", ""),
            "type": props.get("type", ""),
            "image_path": props.get("image_path", "")
        })
    return retrieved_docs

@router.post("/{name}/chat", response_model=ChatResponse)
async def chat_with_collection(name: str, request: ChatRequest):
    """
    Chat with documents in a collection (non-streaming).
    Returns response with inline citations and retrieved documents.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
        
    try:
        session_id = request.session_id or "default"
        thread_id = f"{name}:{session_id}"
        
        config = {"configurable": {"thread_id": thread_id}}
        result = rag.graph.invoke(
            {"query": request.message, "collection_name": name},
            config=config
        )
        
        response_text = ""
        if result.get("messages"):
            last_message = result["messages"][-1]
            response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        return ChatResponse(
            response=response_text,
            retrieved_documents=_format_retrieved_docs(result)
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{name}/chat/stream")
async def stream_chat_with_collection(name: str, request: ChatRequest):
    """
    Stream the RAG response token-by-token using Server-Sent Events (SSE).
    """
    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
        
    session_id = request.session_id or 'default'
    thread_id = f"{name}:{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # Capture trace_id once the OTel span is active
            trace_id = None

            async for event in rag.graph.astream_events(
                {
                    "query": request.message,
                    "collection_name": name,
                },
                config=config,
                version='v2'
            ):
                # Grab trace_id from the first event (span context is now active)
                if trace_id is None:
                    trace_id = get_current_trace_id()

                kind = event['event']

                if kind == "on_chain_end" and event["name"] == "retriever":
                    docs = event["data"]['output'].get("retrieved_documents", [])
                    formatted_docs = _format_retrieved_docs({"retrieved_documents": docs})
                    yield f"data: {json.dumps({'type': 'docs', 'documents': formatted_docs})}\n\n"

                elif kind == "on_chat_model_stream":
                    # Only stream tokens from the final generation nodes (not router/rewriter)
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if node_name in ("rag_generator", "general_llm"):
                        chunk_content = event['data']['chunk'].content
                        if chunk_content:
                            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_content})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'trace_id': trace_id})}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")
