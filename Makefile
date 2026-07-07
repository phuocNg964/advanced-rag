.PHONY: test check

# Run unit tests (no running stack required)
test:
	uv run pytest -q tests/

# Compile-check all Python files for syntax errors + run linter
check:
	uv run python -m compileall -q src tests main.py scripts/smoke_test.py scripts/reingest_doclingpapersv2_with_stats.py
	uv run ruff check src tests main.py scripts
