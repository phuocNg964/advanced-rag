import weaviate
from weaviate.classes.query import Rerank, Filter, MetadataQuery
from typing import List, Optional
from opentelemetry import trace
from src.components.reranker import get_reranker
from src.core.logger import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

try:
    from openinference.semconv.trace import SpanAttributes

    SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
except ImportError:
    SPAN_KIND = "openinference.span.kind"


def resolve_reranker_mode(mode: str) -> str:
    normalized = mode.lower()
    if normalized in {"weaviate", "app", "none"}:
        return normalized

    logger.warning(
        "Invalid RERANKER_MODE '%s'. Falling back to 'weaviate'.",
        mode,
    )
    return "weaviate"


def _build_search_filter(metadata: Optional[dict]):
    if not metadata:
        return None

    filter_list = []
    for key, value in metadata.items():
        if isinstance(value, list):
            filter_list.append(Filter.by_property(key).contains_any(value))
        else:
            filter_list.append(Filter.by_property(key).equal(value))
    return Filter.any_of(filter_list)


def _weaviate_rerank(query: str, reranker_mode: str, top_k_reranker: int):
    if reranker_mode == "weaviate" and top_k_reranker:
        return Rerank(prop="text", query=query)
    return None


def _apply_reranking(query: str, objects: list, top_k_reranker: int, settings, reranker_mode: str):
    if not top_k_reranker:
        return objects
    if reranker_mode == "app":
        return get_reranker(settings).rerank(query, objects, top_k_reranker)
    return objects[:top_k_reranker]


def configured_reranker_model_name(reranker_mode: str, settings) -> str:
    if reranker_mode == "app":
        return settings.app_reranker_model
    if reranker_mode == "weaviate":
        return settings.weaviate_reranker_model
    return "none"


def _span_reranker_model_name(
    reranker_mode: str, top_k_reranker: int, settings
) -> str:
    if top_k_reranker <= 0 or reranker_mode == "none":
        return "none"
    return configured_reranker_model_name(reranker_mode, settings)


def retrieve(
    query: str,
    collection_name: str,
    metadata: Optional[dict] = None,
    top_k: int = 20,
    alpha: float = 0.5,
    top_k_reranker: int = 5,
    client: Optional[weaviate.WeaviateClient] = None,
    raise_errors: bool = False,
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
    with tracer.start_as_current_span("retriever.subquery") as span:
        span.set_attribute(SPAN_KIND, "RETRIEVER")
        span.set_attribute("retriever.query", query)
        span.set_attribute("retriever.alpha", alpha)

        owns_client = client is None
        try:
            settings = get_settings()
            reranker_mode = resolve_reranker_mode(settings.reranker_mode)
            reranker_model_name = _span_reranker_model_name(
                reranker_mode, top_k_reranker, settings
            )
            span.set_attribute(
                "reranker.model_name",
                reranker_model_name,
            )

            if owns_client:
                client = weaviate.connect_to_local(
                    host=settings.weaviate_host,
                    port=settings.weaviate_http_port,
                    grpc_port=settings.weaviate_grpc_port,
                )

            collection = client.collections.get(collection_name)

            search_filter = _build_search_filter(metadata)

            results = collection.query.hybrid(
                query=query,
                filters=search_filter,
                alpha=alpha,
                limit=top_k,
                rerank=_weaviate_rerank(query, reranker_mode, top_k_reranker),
                return_metadata=MetadataQuery(score=True, distance=True),
            )

            span.set_attribute("retriever.initial_search", len(results.objects))

            final_results = _apply_reranking(
                query,
                results.objects,
                top_k_reranker,
                settings,
                reranker_mode,
            )

            reranked_documents = (
                len(final_results) if reranker_model_name != "none" else 0
            )
            span.set_attribute("reranker.reranked_documents", reranked_documents)

            logger.info(f"Retrieved {len(final_results)} results successfully")
            return final_results

        except Exception as e:
            span.set_attribute("retriever.error_type", type(e).__name__)
            span.set_attribute("retriever.error", str(e))
            span.record_exception(e)
            logger.error(f"Retrieval failed: {e}")
            if raise_errors:
                raise
            return []

        finally:
            if owns_client and client is not None:
                client.close()
