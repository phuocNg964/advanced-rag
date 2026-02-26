import weaviate
from weaviate.classes.query import Rerank, Filter, MetadataQuery
from typing import List, Optional
from src.logging_config import get_logger
from src.config import get_settings

logger = get_logger(__name__)

def retrieve(
    query: str,
    collection_name: str,
    metadata: Optional[dict] = None,
    top_k: int = 20,
    alpha: float = 0.5,
    top_k_reranker: int = 5,
    client: Optional[weaviate.WeaviateClient] = None,
    ) -> List:
    """Perform hybrid search with optional reranking and metadata filtering.
    
    Args:
        query: Search query string
        collection_name: Weaviate collection name
        metadata: Optional metadata filters
        top_k: Number of results before reranking
        alpha: Hybrid search weight (0=keyword, 1=vector)
        top_k_reranker: Number of results after reranking
        client: Optional Weaviate client for connection reuse.
                If not provided, a new connection is created and closed after the call.

    Returns:
        List of retrieved document objects, or [] on error
    """
    owns_client = client is None
    try:
        if owns_client:
            settings = get_settings()
            client = weaviate.connect_to_local(
                host=settings.weaviate_host,
                port=settings.weaviate_http_port, 
                grpc_port=settings.weaviate_grpc_port
            )
            
        collection = client.collections.get(collection_name)
        
        # Metadata filtering 
        search_filter = None
        if metadata:
            filter_list = [Filter.by_property(key).contains_any([value]) for key, value in metadata.items()]
            search_filter = Filter.any_of(filter_list)
        
        # Hybrid search
        results = collection.query.hybrid(
            query=query,
            filters=search_filter,
            alpha=alpha,
            limit=top_k,
            rerank=Rerank(
                prop='text',
                query=query
            ) if top_k_reranker else None,
            return_metadata=MetadataQuery(score=True, distance=True),
        )
        final_results = results.objects[:top_k_reranker] if top_k_reranker else results.objects

        logger.info(f"Retrieved {len(final_results)} results successfully")
        return final_results
    
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []
    
    finally:
        if owns_client and client is not None:
            client.close()
