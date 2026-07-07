from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.api.schemas.collections import CollectionCreate
from src.api.validation import collection_name_or_422, document_name_or_422
from src.core.database import CollectionService, CollectionServiceError
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("")
async def create_collection(request: CollectionCreate):
    """Create a new Weaviate collection."""
    collection_name = collection_name_or_422(request.name)
    try:
        service = CollectionService()
        message = service.create(collection_name)
        return {"message": message}
    except CollectionServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to create collection")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{name}")
async def delete_collection(name: str, background_tasks: BackgroundTasks):
    """Delete a Weaviate collection."""
    collection_name = collection_name_or_422(name)
    try:
        service = CollectionService()
        message = service.delete_collection(collection_name)
        # Offload slow disk I/O (deleting thousands of images/text files) to the background
        background_tasks.add_task(service.delete_collection_files, collection_name)
        return {"message": message}
    except CollectionServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to delete collection")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("")
async def list_collections():
    """List all Weaviate collections."""
    try:
        service = CollectionService()
        collections = service.get_all_collections()
        return {"collections": list(collections.keys())}
    except Exception as exc:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{collection_name}/documents")
async def list_documents(collection_name: str):
    """List all documents in a collection."""
    collection_name = collection_name_or_422(collection_name)
    try:
        service = CollectionService()
        documents = service.get_documents(collection_name)
        return {"documents": documents}
    except Exception as exc:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{collection_name}/documents/{document_name}")
async def delete_document(collection_name: str, document_name: str):
    """Delete a document from a collection."""
    collection_name = collection_name_or_422(collection_name)
    document_name = document_name_or_422(document_name)
    try:
        service = CollectionService()
        message = service.delete_document(collection_name, document_name)
        return {"message": message}
    except CollectionServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to delete document")
        raise HTTPException(status_code=500, detail=str(exc))
