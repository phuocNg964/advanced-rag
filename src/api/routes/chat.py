from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Will be initialized by main.py
rag = None

from src.api.schemas.chat import ChatRequest, ChatResponse


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

        result = await rag.chat(name, request.message, session_id)

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
    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    session_id = request.session_id or "default"

    return StreamingResponse(
        rag.stream_chat(name, request.message, session_id),
        media_type="text/event-stream",
    )


@router.get("/{name}/chat/history")
async def get_chat_history(name: str):
    """
    Retrieve the chat history for a given collection.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        # Enforcing single conversation per collection via "default" session
        history = await rag.get_history(name, session_id="default")
        return {"history": history}
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}/chat/history")
async def delete_chat_history(name: str):
    """
    Delete the chat history for a given collection.
    """
    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        # Enforcing single conversation per collection via "default" session
        success = await rag.clear_history(name, session_id="default")
        if success:
            return {"message": "Chat history deleted successfully."}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to clear history from database."
            )
    except Exception as e:
        logger.error(f"Failed to delete history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
