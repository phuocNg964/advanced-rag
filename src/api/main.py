from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from opentelemetry import context as otel_context
from opentelemetry.propagate import extract

from src.core.logger import get_logger, setup_logging
from src.core.config import get_settings
from src.components.reranker import get_reranker
from src.components.docling_parser import warmup_docling
from src.components.retriever import resolve_reranker_mode
from src.core.weaviate_client import (
    init_weaviate,
    close_weaviate,
    is_weaviate_connected,
)
from src.core.job_store import init_ingestion_job_store

from src.api.routes import collections, documents, chat
from src.core.telemetry import init_tracing

logger = get_logger(__name__)
settings = get_settings()

# Ensure data directories exist (they're gitignored, so a fresh clone won't have them)
settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
settings.data_processed_dir.mkdir(parents=True, exist_ok=True)

# Single RAG instance for all collections. Created during lifespan so startup
# failures can be surfaced through /ready instead of import-time crashes.
rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global rag
    setup_logging()
    app.state.rag_ready = False
    app.state.weaviate_ready = False
    app.state.docling_ready = not settings.warmup_docling
    app.state.job_store_ready = False
    app.state.reranker_ready = not (
        settings.warmup_reranker
        and resolve_reranker_mode(settings.reranker_mode) == "app"
    )

    # Initialize Phoenix tracing (must happen before LangChain imports execute)
    init_tracing()

    logger.info("Starting RAG API server")

    # Initialize Persistent RAG
    try:
        from src.agentic_rag.agent_workflow import AgenticRAG

        rag = AgenticRAG()
        chat.rag = rag
        await rag.setup()
        app.state.rag_ready = True
    except Exception as e:
        logger.error(f"Failed to setup RAG: {e}")

    try:
        # Initialize Persistent Weaviate connection
        init_weaviate()
        app.state.weaviate_ready = is_weaviate_connected()
    except Exception as e:
        logger.error(f"Failed to setup Weaviate: {e}")

    try:
        init_ingestion_job_store(settings)
        app.state.job_store_ready = True
    except Exception as e:
        logger.error(f"Failed to setup ingestion job store: {e}")

    if settings.warmup_docling:
        try:
            logger.info("Warming up Docling PDF parser")
            await asyncio.to_thread(warmup_docling, False)
            app.state.docling_ready = True
            logger.info("Docling PDF parser warmup complete")
        except Exception as e:
            logger.error(f"Failed to warm up Docling PDF parser: {e}")
    else:
        logger.info("Docling PDF parser warmup skipped")

    if settings.warmup_reranker and resolve_reranker_mode(settings.reranker_mode) == "app":
        try:
            logger.info("Warming up app reranker")
            await asyncio.to_thread(get_reranker(settings).warmup)
            app.state.reranker_ready = True
            logger.info("App reranker warmup complete")
        except Exception as e:
            logger.error(f"Failed to warm up app reranker: {e}")
    else:
        logger.info("App reranker warmup skipped")

    yield

    # Cleanup
    if rag is not None:
        await rag.close()
    close_weaviate()
    logger.info("Shutting down RAG API server")


app = FastAPI(
    title="Agentic RAG API",
    description="API for document ingestion and RAG-based chat",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_trace_context(request, call_next):
    token = otel_context.attach(extract(dict(request.headers)))
    try:
        return await call_next(request)
    finally:
        otel_context.detach(token)


# Include Routers
app.include_router(collections.router, prefix="/collections", tags=["Collections"])
app.include_router(documents.router, prefix="/collections", tags=["Documents"])
app.include_router(chat.router, prefix="/collections", tags=["Chat"])


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/ready", tags=["System"])
async def readiness_check():
    dependencies = {
        "rag": bool(
            getattr(app.state, "rag_ready", False)
            and rag is not None
            and rag.graph is not None
        ),
        "weaviate": is_weaviate_connected(),
        "docling": bool(getattr(app.state, "docling_ready", False)),
        "job_store": bool(getattr(app.state, "job_store_ready", False)),
        "reranker": bool(getattr(app.state, "reranker_ready", False)),
    }
    ready = all(dependencies.values())
    payload = {
        "status": "ready" if ready else "not_ready",
        "version": "1.0.0",
        "dependencies": dependencies,
    }
    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


# Mount processed data directory for serving citation images
app.mount(
    "/data/processed",
    StaticFiles(directory=settings.base_dir / "data" / "processed"),
    name="processed",
)

# Mount raw data directory for serving source PDFs
app.mount(
    "/data/raw", StaticFiles(directory=settings.base_dir / "data" / "raw"), name="raw"
)

# Static Files (MUST BE LAST - after all API routes)
app.mount(
    "/", StaticFiles(directory=settings.base_dir / "static", html=True), name="static"
)
