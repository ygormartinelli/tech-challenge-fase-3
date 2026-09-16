.PHONY: install lint test generate validate eda notebook train optimize evaluate benchmark publish pipeline run compose airflow smoke api-benchmark

install:
	uv sync --frozen --all-groups

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

test:
	uv run --frozen pytest

generate validate eda train optimize evaluate benchmark publish:
	uv run --frozen python -m techchallenge_fase3.pipelines.$@

pipeline:
	$(MAKE) validate
	$(MAKE) train
	$(MAKE) optimize
	$(MAKE) evaluate
	$(MAKE) benchmark
	$(MAKE) publish

notebook:
	uv run --frozen python scripts/build_notebook.py

run:
	uv run --frozen uvicorn techchallenge_fase3.api.main:app --host 127.0.0.1

compose:
	docker compose --env-file .env -f docker/docker-compose.yml up --build -d --wait

airflow:
	docker compose --env-file .env -f docker/docker-compose.yml --profile airflow up --build -d --wait

smoke:
	uv run --frozen python scripts/verify_stack.py

api-benchmark:
	uv run --frozen python -m techchallenge_fase3.pipelines.api_benchmark
