# Contributing

Thank you for your interest in contributing to **multimodal-agentic-knowledge-base**.

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- Docker Desktop (for integration testing against live services)

## Setup

```bash
git clone https://github.com/phuocNg964/multimodal-agentic-knowledge-base.git
cd multimodal-agentic-knowledge-base
uv sync --frozen
```

## Running Tests

Unit tests require no running services:

```bash
make test
# or: uv run pytest -q tests/
```

Syntax + lint check:

```bash
make check
```

End-to-end smoke test (requires Docker stack running):

```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --pdf path/to/sample.pdf
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting. Run `make check` before submitting a PR. No configuration changes are needed — ruff picks up the project defaults.

## Project Layout

```
src/api/          FastAPI routes and schemas
src/agentic_rag/  LangGraph agent workflow
src/components/   Docling parser, ingestion pipeline, retriever
src/core/         Config, database, job store, logging, telemetry
src/models/       LLM factory and provider adapters
src/prompts/      All prompt templates
configs/model/    models.yaml — LLM role configuration
static/           Browser UI (served by FastAPI)
tests/            Unit tests (no running stack required)
scripts/          Deployment smoke test
evals/            Evaluation scripts (requires dev deps: uv sync --frozen)
```

## Pull Request Checklist

- [ ] `make test` passes
- [ ] `make check` passes (no ruff errors)
- [ ] New behaviour is covered by a test in `tests/` where feasible
- [ ] If you changed `models.yaml` structure, update `src/core/model_config.py` accordingly
- [ ] If you changed the API schema, update the relevant section in README

## Reporting Issues

Open a GitHub issue with:
1. Your OS and Docker Desktop version
2. The full error output (from `docker compose logs api` or terminal)
3. Which profile you ran (`cpu` / `gpu`)
4. A sanitised copy of your `.env` (redact API keys)
