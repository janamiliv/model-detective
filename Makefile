# All targets go through uv so the lockfile is the single source of truth.
# On Windows run from Git Bash or PowerShell with GNU make on PATH.

.PHONY: setup lint format test test-all quantize collect analyze figures clean

setup:            ## create venv, install locked deps, install git hooks
	uv sync
	uv run pre-commit install

lint:             ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

format:           ## auto-fix lint and format
	uv run ruff check --fix .
	uv run ruff format .

test:             ## unit tests only (no GPU, no models)
	uv run pytest -m "not gpu and not slow"

test-all:         ## everything, including gpu/slow
	uv run pytest

quantize:         ## Phase 2: HF -> GGUF -> llama-quantize (not implemented yet)
	@echo "not implemented yet (Phase 2)"

collect:          ## Phase 3: sample pools (not implemented yet)
	@echo "not implemented yet (Phase 3)"

analyze:          ## Phase 4/5: detectors, power curves, harm (not implemented yet)
	@echo "not implemented yet (Phase 4)"

figures:          ## regenerate results/ figures (not implemented yet)
	@echo "not implemented yet (Phase 4/5)"

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
