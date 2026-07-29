.PHONY: setup dev test demo benchmark report lint typecheck security reproduce

setup:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1

dev:
	@echo "dev is unavailable until the web application phase is implemented."
	@exit 1

test:
	uv run pytest

demo:
	@echo "demo is unavailable until the demonstration phase is implemented."
	@exit 1

benchmark:
	@echo "benchmark is unavailable until the benchmark phase is implemented."
	@exit 1

report:
	@echo "report is unavailable until the reporting phase is implemented."
	@exit 1

lint:
	uv run ruff check .

typecheck:
	uv run pyright

security:
	@echo "security scanning is unavailable until the security tooling phase is implemented."
	@exit 1

reproduce:
	@echo "reproduce is unavailable until the reproducibility pipeline phase is implemented."
	@exit 1

