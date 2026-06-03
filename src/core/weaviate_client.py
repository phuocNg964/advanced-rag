import weaviate
from typing import Optional
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

_global_client: Optional[weaviate.WeaviateClient] = None


def init_weaviate():
    """Explicitly initialize the Weaviate connection (used in lifespan)."""
    global _global_client
    settings = get_settings()
    if _global_client is None or not _global_client.is_connected():
        logger.info("Connecting to global Weaviate instance...")
        _global_client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port,
        )


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Get the active Weaviate client."""
    if _global_client is None or not _global_client.is_connected():
        logger.warning(
            "Weaviate client wasn't initialized! Initializing now as fallback."
        )
        init_weaviate()
    return _global_client


def close_weaviate():
    """Explicitly close the Weaviate connection (used in lifespan)."""
    global _global_client
    if _global_client is not None:
        logger.info("Closing global Weaviate instance...")
        _global_client.close()
        _global_client = None
