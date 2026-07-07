FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    HF_HUB_DISABLE_XET=1 \
    DOCLING_ARTIFACTS_PATH=/opt/docling/models \
    APP_RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# Runtime libraries needed by Docling/PyTorch/OpenCV image processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install locked production dependencies. The PyTorch CPU index is configured in pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Cache the Docling models used by src.components.docling_parser so runtime can stay offline.
RUN docling-tools models download \
    --output-dir ${DOCLING_ARTIFACTS_PATH} \
    layout tableformer easyocr

# Cache the default CPU app-level reranker so runtime can stay offline.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${APP_RERANKER_MODEL}', device='cpu')"

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "main.py"]
