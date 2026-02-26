"""
FastAPI endpoints for Agentic RAG system.
Provides collection management, document ingestion, and chat capabilities.
"""
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ingest import IngestService
from src.rag_workflow import AgenticRAG
from src.collection_service import CollectionService
from src.logging_config import get_logger, setup_logging
from src.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# In-memory job status storage
jobs: Dict[str, Dict[str, Any]] = {}

# Max file size for uploads (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    setup_logging()
    logger.info("Starting RAG API server")
    yield
    logger.info("Shutting down RAG API server")


app = FastAPI(
    title="Agentic RAG API",
    description="API for document ingestion and RAG-based chat",
    version="1.0.0",
    lifespan=lifespan
)

# Mount processed data directory for serving citation images
app.mount("/data/processed", StaticFiles(directory=settings.base_dir / "data" / "processed"), name="processed")

# Mount raw data directory for serving source PDFs
app.mount("/data/raw", StaticFiles(directory=settings.base_dir / "data" / "raw"), name="raw")

# ========================
# Pydantic Models
# ========================

class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, max_length=100)


class ChatResponse(BaseModel):
    response: str
    retrieved_documents: List[Dict[str, Any]]


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# ========================
# Collection Endpoints
# ========================

@app.post("/collections")
async def create_collection(request: CollectionCreate):
    """Create a new Weaviate collection."""
    try:
        with CollectionService() as service:
            message = service.create(request.name)
        return {"message": message}
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete a Weaviate collection."""
    try:
        with CollectionService() as service:
            message = service.delete_collection(name)
        return {"message": message}
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """List all Weaviate collections."""
    try:
        with CollectionService() as service:
            collections = service.get_all_collections()
        return {"collections": list(collections.keys())}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_name}/documents")
async def list_documents(collection_name: str):
    """List all documents in a collection."""
    try:
        with CollectionService() as service:
            documents = service.get_documents(collection_name)
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}/documents/{document_name}")
async def delete_document(collection_name: str, document_name: str):
    """Delete a document from a collection."""
    try:
        with CollectionService() as service:
            message = service.delete_document(collection_name, document_name)
        return {"message": message}
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Document Ingestion Endpoints
# ========================

def _do_ingestion(file_path: Path, collection_name: str):
    """Synchronous ingestion work - runs in thread pool."""
    with IngestService() as service:
        service.ingest(file_name=file_path.name, collection_name=collection_name)


async def run_ingestion_job_async(job_id: str, file_path: Path, collection_name: str):
    """Background task for document ingestion - runs in thread pool to avoid blocking."""
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting ingestion job {job_id} for {file_path}")
        
        # Run sync code in thread pool (doesn't block other API calls)
        await asyncio.to_thread(_do_ingestion, file_path, collection_name)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = f"Successfully ingested {file_path.name}"
        logger.info(f"Completed ingestion job {job_id}")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)
        logger.error(f"Ingestion job {job_id} failed: {e}")


@app.post("/collections/{name}/documents", response_model=JobResponse)
async def upload_document(
    name: str,
    file: UploadFile = File(...)
):
    """
    Upload a PDF document and ingest it into the collection.
    The ingestion runs asynchronously in the background.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )
    
    # Save uploaded file to data/raw/{collection_name}/
    raw_dir = settings.base_dir / "data" / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Create job and start background task
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "file": file.filename,
        "collection": name,
        "message": None
    }
    
    # Use asyncio.create_task for async background work
    asyncio.create_task(run_ingestion_job_async(job_id, file_path, name))
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion job started for {file.filename}"
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Check the status of an ingestion job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message")
    )


# ========================
# Chat Endpoints
# ========================

# Single RAG instance for all collections
rag = AgenticRAG()


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


@app.post("/collections/{name}/chat", response_model=ChatResponse)
async def chat_with_collection(name: str, request: ChatRequest):
    """
    Chat with documents in a collection (non-streaming).
    Returns response with inline citations and retrieved documents.
    
    Uses session_id for conversation memory - same session_id will maintain context.
    """
    try:
        # Build thread_id from collection + session for isolated conversations
        session_id = request.session_id or "default"
        thread_id = f"{name}:{session_id}"
        
        # Invoke RAG workflow with collection_name in state
        config = {"configurable": {"thread_id": thread_id}}
        result = rag.graph.invoke(
            {"query": request.message, "collection_name": name},
            config=config
        )
        
        # Extract response and documents
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


# ========================
# Static Files (MUST BE LAST - after all API routes)
# ========================
app.mount("/", StaticFiles(directory=settings.base_dir / "static", html=True), name="static")


