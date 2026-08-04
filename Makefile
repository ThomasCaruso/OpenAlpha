.PHONY: help setup test test-smoke lint typecheck verify check reports clean

help:
	@echo "OpenAlpha - preregistered research on financial foundation models"
	@echo ""
	@echo "  make setup       Install pinned Python and Node dependencies"
	@echo "  make check       Everything CI runs: tests, lint, types, environment"
	@echo ""
	@echo "  make test        Full suite (~1,549 tests; no GPU or credentials needed)"
	@echo "  make test-smoke  Repository and deployment guards only (fast)"
	@echo "  make lint        ruff"
	@echo "  make typecheck   pyright"
	@echo "  make verify      Environment, sealed specifications, published artifacts"
	@echo "  make reports     Where the completed research reports live"
	@echo ""
	@echo "Executing a study needs a Modal account and object-store credentials;"
	@echo "see docs/BRIDGE_MODAL_DEPLOYMENT.md. No target here starts a run."

setup:
	uv sync --locked --group dev
	npm ci

test:
	uv run --group sentinel-phase3 pytest -q

test-smoke:
	uv run pytest tests/smoke -q

lint:
	uv run ruff check .

typecheck:
	uv run pyright

verify:
	uv run python scripts/verify_environment.py
	uv run python scripts/verify_specifications.py
	uv run python scripts/verify_artifacts.py

check: test lint typecheck verify

reports:
	@echo "Completed research reports:"
	@echo "  research/reports/README.md                    index of all three studies"
	@echo "  research/reports/kronos-structural-validity/  studies 1 and 2"
	@echo "  research/reports/kronos-zero-shot-benchmark/  study 3"

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
