import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.components.ingestion import IngestService
from src.core.database import CollectionService
from src.core.logger import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# In-memory job status storage
jobs: Dict[str, Dict[str, Any]] = {}
MAX_FILE_SIZE = 100 * 1024 * 1024

from src.api.schemas.documents import JobResponse


def _do_ingestion(file_path: Path, collection_name: str):
    """Synchronous ingestion work - runs in thread pool."""
    with IngestService() as service:
        service.ingest(file_name=file_path.name, collection_name=collection_name)


async def run_ingestion_job_async(job_id: str, file_path: Path, collection_name: str):
    """Background task for document ingestion - runs in thread pool to avoid blocking."""
    from opentelemetry import trace
    from src.core.telemetry import get_current_trace_id
    tracer = trace.get_tracer(__name__)
    
    # Wrap the entire background job in a root span
    with tracer.start_as_current_span("ingestion_job") as span:
        trace_id = get_current_trace_id()
        jobs[job_id]["trace_id"] = trace_id
        
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
            logger.error(f"Ingestion job {job_id} failed: {e}. Initiating rollback...")
        
            # Rollback: purge partially inserted chunks and cleanly remove the file so user can try again
            try:
                coll_service = CollectionService()
                coll_service.delete_document(collection_name, file_path.name)
                logger.info(f"Rollback complete for {file_path.name}")
            except Exception as rollback_e:
                logger.error(f"Rollback failed for {file_path.name}: {rollback_e}")


@router.post("/{name}/documents", response_model=JobResponse)
async def upload_document(
    name: str,
    file: UploadFile = File(...)
):
    """
    Upload a PDF document and ingest it into the collection.
    The ingestion runs asynchronously in the background.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )
    
    raw_dir = settings.base_dir / "data" / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file.filename
    
    try:
        # Pre-cleanup: Prevent the "Duplicate Chunk" bug by deleting previous vectors/files safely
        coll_service = CollectionService()
        coll_service.delete_document(name, file.filename)
            
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "file": file.filename,
        "collection": name,
        "message": None
    }
    
    asyncio.create_task(run_ingestion_job_async(job_id, file_path, name))
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion job started for {file.filename}"
    )

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Check the status of an ingestion job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
        trace_id=job.get("trace_id")
    )
