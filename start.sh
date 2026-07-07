#!/bin/bash
set -e

PROFILE="${1:-cpu}"
OBS_FLAG="${2:-}"

if [ "$PROFILE" != "cpu" ] && [ "$PROFILE" != "gpu" ]; then
    echo "Usage: ./start.sh [cpu|gpu] [--observability]"
    exit 1
fi

if [ -n "$OBS_FLAG" ] && [ "$OBS_FLAG" != "--observability" ]; then
    echo "Usage: ./start.sh [cpu|gpu] [--observability]"
    exit 1
fi

echo "========================================="
echo "Starting Agentic RAG Local Environment..."
echo "========================================="

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker was not found. Install Docker Desktop or Docker Engine, then rerun this script."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose was not found or Docker is not running. Start Docker, then rerun this script."
    exit 1
fi

echo ""
echo "[1/3] Creating necessary directories..."
mkdir -p data/raw
mkdir -p data/processed

echo ""
echo "[2/3] Checking environment configuration..."
if [ ! -f .env ]; then
    echo "No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "======================================================="
    echo "Created .env. Please edit it and set GROQ_API_KEY, then rerun this script."
    echo "======================================================="
    exit 1
else
    echo ".env file found."
fi

if grep -q "your_groq_api_key_here" .env; then
    echo "======================================================="
    echo "GROQ_API_KEY is still the placeholder value in .env."
    echo "Please set a real key, then rerun this script."
    echo "======================================================="
    exit 1
fi

echo ""
echo "[3/3] Starting Docker containers (this may take a while on first run)..."
echo "    Using $PROFILE profile."

COMPOSE_ARGS=(--profile "$PROFILE" --profile production)
if [ "$OBS_FLAG" = "--observability" ]; then
    export DOCKER_PHOENIX_COLLECTOR_ENDPOINT="${DOCKER_PHOENIX_COLLECTOR_ENDPOINT:-http://phoenix:4317}"
    COMPOSE_ARGS+=(--profile observability)
    echo "    Phoenix observability enabled."
fi

docker compose "${COMPOSE_ARGS[@]}" up -d --build

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "Your Agentic RAG system is starting up in the background."
echo "It may take a minute or two for all AI models to load."
echo ""
echo "Access the UI and API here: http://localhost:8000"
if [ "$OBS_FLAG" = "--observability" ]; then
    echo "Phoenix tracing UI: http://localhost:6006"
fi
echo "========================================="
