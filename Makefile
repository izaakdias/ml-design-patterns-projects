.PHONY: lint test check train

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

test:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q

check: lint test

train:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m ml_patterns_lab train all
