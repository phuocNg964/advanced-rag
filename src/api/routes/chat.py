from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas.chat import ChatRequest, ChatResponse
from src.api.validation import collection_name_or_422
from src.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Will be initialized by main.py
rag = None


def _require_rag():
    if not rag or getattr(rag, "graph", None) is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    return rag


@router.post("/{name}/chat", response_model=ChatResponse)
async def chat_with_collection(name: str, request: ChatRequest):
    """
    Chat with documents in a collection (non-streaming).
    Returns response with inline citations and retrieved documents.
    """
    active_rag = _require_rag()
    collection_name = collection_name_or_422(name)
    try:
        session_id = request.session_id or "default"

        result = await active_rag.chat(collection_name, request.message, session_id)

        return ChatResponse(
            response=result["response"],
            retrieved_documents=result["retrieved_documents"],
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{name}/chat/stream")
async def stream_chat_with_collection(name: str, request: ChatRequest):
    """
    Stream the RAG response token-by-token using Server-Sent Events (SSE).
    """
    active_rag = _require_rag()
    collection_name = collection_name_or_422(name)
    session_id = request.session_id or "default"

    return StreamingResponse(
        active_rag.stream_chat(collection_name, request.message, session_id),
        media_type="text/event-stream",
    )


@router.get("/{name}/chat/history")
async def get_chat_history(name: str):
    """
    Retrieve the chat history for a given collection.
    """
    active_rag = _require_rag()
    collection_name = collection_name_or_422(name)
    try:
        # Enforcing single conversation per collection via "default" session
        history = await active_rag.get_history(collection_name, session_id="default")
        return {"history": history}
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}/chat/history")
async def delete_chat_history(name: str):
    """
    Delete the chat history for a given collection.
    """
    active_rag = _require_rag()
    collection_name = collection_name_or_422(name)
    try:
        # Enforcing single conversation per collection via "default" session
        await active_rag.clear_history(collection_name, session_id="default")
        return {"message": "Chat history deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
