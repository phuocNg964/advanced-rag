@echo off
echo =========================================
echo Starting Agentic RAG Local Environment...
echo =========================================

echo.
echo [1/3] Creating necessary directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"

echo.
echo [2/3] Checking environment configuration...
if not exist ".env" (
    echo No .env file found. Copying from .env.example...
    copy .env.example .env
    echo =======================================================
    echo IMPORTANT: Please edit the .env file to add your API keys
    echo before using the ingestion and chat features!
    echo =======================================================
) else (
    echo .env file found.
)

echo.
echo [3/3] Starting Docker containers (this may take a while on first run)...
echo     Using CPU profile. For GPU, edit this script to use --profile gpu
docker compose --profile cpu --profile production up -d --build

echo.
echo =========================================
echo Setup Complete!
echo =========================================
echo Your Agentic RAG system is starting up in the background.
echo It may take a minute or two for all AI models to load.
echo.
echo Access the UI and API here: http://localhost:8000
echo =========================================
pause
