from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.core.database import CollectionService
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

from src.api.schemas.collections import CollectionCreate


@router.post("")
async def create_collection(request: CollectionCreate):
    """Create a new Weaviate collection."""
    try:
        service = CollectionService()
        message = service.create(request.name)
        return {"message": message}
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@router.delete("/{name}")
async def delete_collection(name: str, background_tasks: BackgroundTasks):
    """Delete a Weaviate collection."""
    try:
        service = CollectionService()
        message = service.delete_collection(name)
        # Offload slow disk I/O (deleting thousands of images/text files) to the background
        background_tasks.add_task(service.delete_collection_files, name)
        return {"message": message}
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_collections():
    """List all Weaviate collections."""
    try:
        service = CollectionService()
        collections = service.get_all_collections()
        return {"collections": list(collections.keys())}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{collection_name}/documents")
async def list_documents(collection_name: str):
    """List all documents in a collection."""
    try:
        service = CollectionService()
        documents = service.get_documents(collection_name)
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{collection_name}/documents/{document_name}")
async def delete_document(collection_name: str, document_name: str):
    """Delete a document from a collection."""
    try:
        service = CollectionService()
        message = service.delete_document(collection_name, document_name)
        return {"message": message}
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
