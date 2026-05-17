from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, max_length=100)

class ChatResponse(BaseModel):
    response: str
    retrieved_documents: List[Dict[str, Any]]
