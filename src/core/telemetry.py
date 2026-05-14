"""
Phoenix observability tracing setup.

Initializes OpenTelemetry tracing with Phoenix as the collector backend.
Auto-instruments all LangChain/LangGraph calls when enabled.
"""
import os
import logging
from dotenv import load_dotenv
from phoenix.otel import register
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Ensure .env is loaded so os.getenv can read it
load_dotenv()

logger = logging.getLogger(__name__)
_tracing_enabled = False

def init_tracing():
    """
    Initialize Phoenix tracing if PHOENIX_COLLECTOR_ENDPOINT is set.

    In local dev: set PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
    In Docker:    set automatically via docker-compose env
    Disabled:     simply don't set the env var
    """
    global _tracing_enabled
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    if not endpoint:
        logger.info("PHOENIX_COLLECTOR_ENDPOINT not set — tracing disabled.")
        return
    try:
        tracer_provider = register(
            project_name="agentic-rag",
            auto_instrument=True,  # Auto-patches langchain-core, openai, etc.
            #span_processor=None if is_dev else BatchSpanProcessor(...)
        )
        _tracing_enabled = True
        logger.info(f"Phoenix tracing enabled → {endpoint}")
    except ImportError:
        logger.warning(
            "Phoenix tracing packages not installed. "
            "Run: pip install arize-phoenix-otel openinference-instrumentation-langchain"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Phoenix tracing: {e}")


def is_tracing_enabled() -> bool:
    """Check if tracing is currently active."""
    return _tracing_enabled


def get_current_trace_id() -> str | None:
    """Extract the current OpenTelemetry trace ID (hex string) if tracing is active."""
    if not _tracing_enabled:
        return None
    try:
        span = otel_trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            return otel_trace.format_trace_id(ctx.trace_id)
    except Exception:
        pass
    return None
