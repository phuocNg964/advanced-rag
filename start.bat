@echo off
setlocal

set PROFILE=%~1
if "%PROFILE%"=="" set PROFILE=cpu
set OBS_PROFILE=

if /I not "%PROFILE%"=="cpu" if /I not "%PROFILE%"=="gpu" goto usage
if not "%~2"=="" (
    if /I "%~2"=="--observability" (
        set OBS_PROFILE=--profile observability
        if "%DOCKER_PHOENIX_COLLECTOR_ENDPOINT%"=="" set DOCKER_PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:4317
    ) else (
        goto usage
    )
)

echo =========================================
echo Starting Agentic RAG Local Environment...
echo =========================================

where docker >nul 2>nul
if not "%ERRORLEVEL%"=="0" (
    echo Docker was not found. Install Docker Desktop, then rerun this script.
    pause
    exit /b 1
)

docker compose version >nul 2>nul
if not "%ERRORLEVEL%"=="0" (
    echo Docker Compose was not found or Docker Desktop is not running. Start Docker, then rerun this script.
    pause
    exit /b 1
)

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
    echo Created .env. Please edit it and set GROQ_API_KEY, then rerun this script.
    echo =======================================================
    pause
    exit /b 1
) else (
    echo .env file found.
)

findstr /C:"your_groq_api_key_here" .env >nul
if "%ERRORLEVEL%"=="0" (
    echo =======================================================
    echo GROQ_API_KEY is still the placeholder value in .env.
    echo Please set a real key, then rerun this script.
    echo =======================================================
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Docker containers (this may take a while on first run)...
echo     Using %PROFILE% profile.
if not "%OBS_PROFILE%"=="" echo     Phoenix observability enabled.
docker compose --profile %PROFILE% --profile production %OBS_PROFILE% up -d --build

echo.
echo =========================================
echo Setup Complete!
echo =========================================
echo Your Agentic RAG system is starting up in the background.
echo It may take a minute or two for all AI models to load.
echo.
echo Access the UI and API here: http://localhost:8000
if not "%OBS_PROFILE%"=="" echo Phoenix tracing UI: http://localhost:6006
echo =========================================
pause
exit /b 0

:usage
echo Usage: start.bat [cpu^|gpu] [--observability]
pause
exit /b 1
