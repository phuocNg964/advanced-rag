import weaviate
import shutil
from pathlib import Path
from weaviate.classes.config import Configure, Property, DataType
from typing import Optional
from weaviate.classes.query import Filter
from src.logging_config import get_logger
from src.config import Settings, get_settings

logger = get_logger(__name__)


class CollectionService:
    """Service for managing Weaviate collections."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client: Optional[weaviate.WeaviateClient] = None

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup on block exit."""
        self.close()
        return False

    def _get_client(self) -> weaviate.WeaviateClient:
        """Get or create Weaviate client using config settings."""
        if self.client is None or not self.client.is_connected():
            self.client = weaviate.connect_to_local(
                host=self.settings.weaviate_host,
                port=self.settings.weaviate_http_port,
                grpc_port=self.settings.weaviate_grpc_port
            )
        return self.client


    def create(self, collection_name: str):
        """Create a new Weaviate collection with standard RAG properties."""
        try:
            self._get_client()
            if self.client.collections.exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return 'Collection already exists'
            
            self.client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                reranker_config=Configure.Reranker.transformers(),
                properties=[
                    Property(name='text', data_type=DataType.TEXT, skip_vectorization=False),
                    Property(name='chunk_id', data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name='type', data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name='source', data_type=DataType.TEXT, skip_vectorization=False),
                    Property(name='image_path', data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name='caption', data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name='page_number', data_type=DataType.INT, skip_vectorization=True),
                ]
            )
            
            # Create collection folders
            raw_dir = self.settings.base_dir / "data" / "raw" / collection_name
            processed_dir = self.settings.base_dir / "data" / "processed" / collection_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Collection '{collection_name}' created successfully")
            return 'Collection created successfully'

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return 'Failed to create collection'


    def delete_collection(self, collection_name: str):
        """Delete a Weaviate collection and its data folders."""
        self._get_client()

        if not self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist")
            return 'Collection does not exist'
        
        self.client.collections.delete(collection_name)
        
        # Delete collection folders
        raw_dir = self.settings.base_dir / "data" / "raw" / collection_name
        processed_dir = self.settings.base_dir / "data" / "processed" / collection_name
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        
        logger.info(f"Collection '{collection_name}' deleted successfully")
        return 'Collection deleted successfully'


    def delete_document(self, collection_name: str, file_document: str):
        """Delete a document's vectors from the collection and its files from disk."""
        try:
            self._get_client()
            
            collection = self.client.collections.get(collection_name)

            collection.data.delete_many(
                where=Filter.by_property('source').like(f"*{file_document}*")
            )
            logger.info(f"Document '{file_document}' deleted from collection")
            
            # Delete files from filesystem
            self._delete_document_files(collection_name, file_document)
            
            return f"Document '{file_document}' deleted successfully"

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return f"Failed to delete document: {e}"

    def _delete_document_files(self, collection_name: str, file_document: str):
        """Delete document files from collection's data folders."""
        # Delete PDF from data/raw/{collection}
        raw_file = self.settings.base_dir / "data" / "raw" / collection_name / file_document
        if raw_file.exists():
            raw_file.unlink()
            logger.info(f"Deleted raw file: {raw_file}")
        
        # Delete processed folder (images/tables extracted from PDF)
        folder_name = file_document.rsplit('.', 1)[0] if '.' in file_document else file_document
        processed_folder = self.settings.base_dir / "data" / "processed" / collection_name / folder_name
        if processed_folder.exists() and processed_folder.is_dir():
            shutil.rmtree(processed_folder)
            logger.info(f"Deleted processed folder: {processed_folder}")

    def get_documents(self, collection_name: str):
        """Get list of documents in collection's data/raw folder."""
        raw_dir = self.settings.base_dir / "data" / "raw" / collection_name
        
        documents = []
        if raw_dir.exists():
            for file_path in raw_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                    documents.append({
                        "filename": file_path.name,
                        "source": str(file_path)
                    })
        
        return documents

    def get_all_collections(self):
        """List all Weaviate collections."""
        self._get_client()
        return self.client.collections.list_all()

    def close(self):
        if self.client:
            self.client.close()