import shutil
import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.api.schemas.documents import JobResponse
from src.api.validation import collection_name_or_422, document_name_or_422
from src.components.ingestion import IngestService
from src.core.database import CollectionService
from src.core.job_store import IngestionJobStore
from src.core.logger import get_logger
from src.core.config import get_settings
from src.core.names import resolve_child_path

logger = get_logger(__name__)
settings = get_settings()
job_store = IngestionJobStore(settings)

router = APIRouter()


def _do_ingestion(file_path: Path, collection_name: str):
    """Synchronous ingestion work - runs in thread pool."""
    service = IngestService()
    return service.ingest(file_name=file_path.name, collection_name=collection_name)


async def _update_job_status(job_id: str, **fields):
    try:
        await asyncio.to_thread(job_store.update_job, job_id, **fields)
    except Exception:
        logger.exception("Failed to update ingestion job %s", job_id)


async def run_ingestion_job_async(job_id: str, file_path: Path, collection_name: str):
    """Background task for document ingestion - runs in thread pool to avoid blocking."""
    from opentelemetry import trace
    from src.core.telemetry import get_current_trace_id

    tracer = trace.get_tracer(__name__)

    # Wrap the entire background job in a root span
    with tracer.start_as_current_span("ingestion_job"):
        trace_id = get_current_trace_id()

        try:
            logger.info("Captured trace %s for ingestion job %s", trace_id, job_id)
            await _update_job_status(
                job_id,
                status="processing",
                trace_id=trace_id,
            )
            logger.info(f"Starting ingestion job {job_id} for {file_path}")

            # Run sync code in thread pool (doesn't block other API calls)
            result = await asyncio.to_thread(_do_ingestion, file_path, collection_name)

            if result.warnings:
                message = f"Successfully ingested {file_path.name} with warnings"
            else:
                message = f"Successfully ingested {file_path.name}"
            await _update_job_status(
                job_id,
                status="completed",
                message=message,
                warnings=result.warnings,
            )
            logger.info(f"Completed ingestion job {job_id}")

        except Exception as e:
            await _update_job_status(
                job_id,
                status="failed",
                message=str(e),
            )
            logger.error(f"Ingestion job {job_id} failed: {e}. Initiating rollback...")

            # Rollback: purge partially inserted chunks and cleanly remove the file so user can try again
            try:
                coll_service = CollectionService()
                coll_service.delete_document(collection_name, file_path.name)
                logger.info(f"Rollback complete for {file_path.name}")
            except Exception as rollback_e:
                logger.error(f"Rollback failed for {file_path.name}: {rollback_e}")


@router.post("/{name}/documents", response_model=JobResponse)
async def upload_document(name: str, file: UploadFile = File(...)):
    """
    Upload a PDF document and ingest it into the collection.
    The ingestion runs asynchronously in the background.
    """
    collection_name = collection_name_or_422(name)
    document_name = document_name_or_422(file.filename or "")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    max_file_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    raw_dir = resolve_child_path(settings.data_raw_dir, collection_name)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = resolve_child_path(raw_dir, document_name)

    try:
        # Pre-cleanup: Prevent the "Duplicate Chunk" bug by deleting previous vectors/files safely
        coll_service = CollectionService()
        coll_service.delete_document(collection_name, document_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    job_id = str(uuid4())
    await asyncio.to_thread(
        job_store.create_job,
        job_id=job_id,
        collection_name=collection_name,
        document_name=document_name,
        status="queued",
        warnings=[],
    )

    asyncio.create_task(run_ingestion_job_async(job_id, file_path, collection_name))

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion job started for {document_name}",
        warnings=[],
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: UUID):
    """Check the status of an ingestion job."""
    job_id_str = str(job_id)
    job = await asyncio.to_thread(job_store.get_job, job_id_str)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job_id_str,
        status=job["status"],
        message=job.get("message"),
        trace_id=job.get("trace_id"),
        warnings=job.get("warnings", []),
    )
