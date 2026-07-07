import shutil

from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from src.core.config import Settings, get_settings
from src.core.logger import get_logger
from src.core.names import (
    resolve_child_path,
    validate_collection_name,
    validate_document_filename,
)
from src.core.weaviate_client import get_weaviate_client

logger = get_logger(__name__)


class CollectionServiceError(RuntimeError):
    """Raised when collection storage operations fail."""


class CollectionService:
    """Service for managing Weaviate collections."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        # Fetch the centrally managed instance
        self.client = get_weaviate_client()

    def create(self, collection_name: str):
        """Create a new Weaviate collection with standard RAG properties."""
        collection_name = validate_collection_name(collection_name)
        if self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            return "Collection already exists"

        create_kwargs = {
            "name": collection_name,
            "vectorizer_config": Configure.Vectorizer.text2vec_transformers(),
            "properties": self._collection_properties(),
        }
        if self.settings.reranker_mode.lower() == "weaviate":
            create_kwargs["reranker_config"] = Configure.Reranker.transformers()

        try:
            self.client.collections.create(**create_kwargs)
        except Exception as exc:
            logger.exception("Failed to create collection '%s'", collection_name)
            raise CollectionServiceError(
                f"Failed to create collection '{collection_name}'"
            ) from exc

        raw_dir, processed_dir = self._collection_dirs(collection_name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Collection '{collection_name}' created successfully")
        return "Collection created successfully"

    def delete_collection(self, collection_name: str):
        """Delete a Weaviate collection and its data folders."""
        collection_name = validate_collection_name(collection_name)

        if not self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist")
            return "Collection does not exist"

        self.client.collections.delete(collection_name)

        logger.info(
            f"Collection '{collection_name}' deleted from Weaviate successfully"
        )
        return "Collection deleted successfully"

    def delete_collection_files(self, collection_name: str):
        """Delete collection folders from disk. Can be run in background."""
        collection_name = validate_collection_name(collection_name)
        raw_dir, processed_dir = self._collection_dirs(collection_name)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        logger.info(f"Collection '{collection_name}' files deleted from disk")

    def delete_document(self, collection_name: str, file_document: str):
        """Delete a document's vectors from the collection and its files from disk."""
        collection_name = validate_collection_name(collection_name)
        file_document = validate_document_filename(file_document)
        try:
            collection = self.client.collections.get(collection_name)

            collection.data.delete_many(
                where=Filter.by_property("source").equal(
                    f"{collection_name}/{file_document}"
                )
            )
            logger.info(f"Document '{file_document}' deleted from collection")

            # Delete files from filesystem
            self._delete_document_files(collection_name, file_document)

            return f"Document '{file_document}' deleted successfully"

        except Exception as exc:
            logger.exception(
                "Failed to delete document '%s' from '%s'",
                file_document,
                collection_name,
            )
            raise CollectionServiceError(
                f"Failed to delete document '{file_document}'"
            ) from exc

    def _delete_document_files(self, collection_name: str, file_document: str):
        """Delete document files from collection's data folders."""
        # Delete PDF from data/raw/{collection}
        raw_dir, processed_dir = self._collection_dirs(collection_name)
        raw_file = resolve_child_path(raw_dir, file_document)
        if raw_file.exists():
            raw_file.unlink()
            logger.info(f"Deleted raw file: {raw_file}")

        # Delete processed folder (images/tables extracted from PDF)
        processed_folder = processed_dir / file_document.rsplit(".", 1)[0]
        if processed_folder.exists() and processed_folder.is_dir():
            shutil.rmtree(processed_folder)
            logger.info(f"Deleted processed folder: {processed_folder}")

    def get_documents(self, collection_name: str):
        """Get list of documents in collection's data/raw folder."""
        collection_name = validate_collection_name(collection_name)
        raw_dir, _processed_dir = self._collection_dirs(collection_name)

        documents = []
        if raw_dir.exists():
            for file_path in raw_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".pdf":
                    documents.append(
                        {"filename": file_path.name, "source": str(file_path)}
                    )

        return documents

    def get_all_collections(self):
        """List all Weaviate collections."""
        return self.client.collections.list_all()

    def _collection_dirs(self, collection_name: str):
        raw_dir = resolve_child_path(self.settings.data_raw_dir, collection_name)
        processed_dir = resolve_child_path(
            self.settings.data_processed_dir, collection_name
        )
        return raw_dir, processed_dir

    @staticmethod
    def _collection_properties() -> list[Property]:
        return [
            Property(name="text", data_type=DataType.TEXT, skip_vectorization=False),
            Property(name="chunk_id", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="type", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="source", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="image_path", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="page_number", data_type=DataType.INT, skip_vectorization=True),
            Property(name="section", data_type=DataType.TEXT, skip_vectorization=True),
        ]
