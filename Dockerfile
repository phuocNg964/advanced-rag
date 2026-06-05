FROM python:3.11-slim

# Install system dependencies for unstructured and pdf processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Allow users to override the PyTorch index (e.g., to empty string for CUDA) if they run the API with GPU
ARG PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# Install python dependencies (timeout increased for stability on slow connections)
COPY requirements.lock .
RUN pip install --no-cache-dir --default-timeout=100 \
    --extra-index-url ${PIP_EXTRA_INDEX_URL} \
    -r requirements.lock

# Pre-download the YOLOX layout detection model (~80MB) at build time.
# Eliminates ~60-90s model download on every container start.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('unstructuredio/yolo_x_layout', 'yolox_l0.05.onnx')"

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "main.py"]
