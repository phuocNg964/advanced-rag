from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.core.logger import get_logger, setup_logging
from src.core.config import get_settings
from src.agent.workflow import AgenticRAG
from src.core.weaviate_client import init_weaviate, close_weaviate

from src.api.routers import collections, documents, chat
from src.core.telemetry import init_tracing

logger = get_logger(__name__)
settings = get_settings()

# Single RAG instance for all collections globally accessible to routers if needed
rag = AgenticRAG()

# Assign the rag instance to chat router directly to maintain state
chat.rag = rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    setup_logging()
    
    # Initialize Phoenix tracing (must happen before LangChain imports execute)
    init_tracing()
    
    logger.info("Starting RAG API server")
    
    # Initialize Persistent RAG
    try:
        await rag.setup()
    except Exception as e:
        logger.error(f"Failed to setup RAG: {e}")
        
    try:
        # Initialize Persistent Weaviate connection
        init_weaviate()
    except Exception as e:
        logger.error(f"Failed to setup Weaviate: {e}")
        
    yield
    
    # Cleanup
    await rag.close()
    close_weaviate()
    logger.info("Shutting down RAG API server")

app = FastAPI(
    title="Agentic RAG API",
    description="API for document ingestion and RAG-based chat",
    version="1.0.0",
    lifespan=lifespan
)

# Include Routers
app.include_router(collections.router, prefix="/collections", tags=["Collections"])
app.include_router(documents.router, prefix="/collections", tags=["Documents"])
app.include_router(chat.router, prefix="/collections", tags=["Chat"])

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Mount processed data directory for serving citation images
app.mount("/data/processed", StaticFiles(directory=settings.base_dir / "data" / "processed"), name="processed")

# Mount raw data directory for serving source PDFs
app.mount("/data/raw", StaticFiles(directory=settings.base_dir / "data" / "raw"), name="raw")

# Static Files (MUST BE LAST - after all API routes)
app.mount("/", StaticFiles(directory=settings.base_dir / "static", html=True), name="static")
