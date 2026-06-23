.PHONY: install test run lint typecheck clean ingest ingest-s3

install:
	pipenv install --dev

test:
	pipenv run pytest tests/ -v

test-coverage:
	pipenv run pytest tests/ -v --cov=src --cov=backend --cov-report=term-missing

run:
	pipenv run python backend/run.py

ingest:
	pipenv run python -m src.cli ingest $(filepath) $(if $(filter yes,$(analyze)),--analyze)

ingest-s3:
	pipenv run python -m src.cli ingest-s3 $(bucket) $(if $(prefix),--prefix $(prefix))

lint:
	@if pipenv run ruff check . 2>/dev/null; then \
		pipenv run ruff check .; \
	else \
		echo "ruff not installed; skipping lint"; \
	fi

typecheck:
	@if pipenv run mypy . 2>/dev/null; then \
		pipenv run mypy .; \
	else \
		echo "mypy not installed; skipping typecheck"; \
	fi

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .coverage htmlcov
