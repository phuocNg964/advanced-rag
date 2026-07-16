COMPOSE ?= docker compose
DEV_COMPOSE ?= docker compose -f docker-compose.yaml -f docker-compose.dev.yaml
GPU_COMPOSE ?= docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml --profile gpu

.PHONY: up build gpu dev down logs restart-api check

# Start the default local CPU stack
up:
	$(COMPOSE) up -d

# Rebuild images and start the default stack
build:
	$(COMPOSE) up -d --build

# Start the GPU override stack
gpu:
	$(GPU_COMPOSE) up -d

# Start with source/config bind mounts for local development
dev:
	$(DEV_COMPOSE) up -d

# Stop containers and release network resources
down:
	$(COMPOSE) down

# Follow API logs
logs:
	$(COMPOSE) logs -f api

# Restart only the API container
restart-api:
	$(COMPOSE) restart api

# Compile-check all Python files for syntax errors + run linter
check:
	uv run python -m compileall -q src evals main.py
	uv run ruff check src evals main.py
