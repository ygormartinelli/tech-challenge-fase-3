.PHONY: install lint test train optimize benchmark api-benchmark run compose airflow

install:
	uv sync --all-groups

lint:
	uv run ruff check .

test:
	uv run pytest

train:
	uv run python -m techchallenge_fase3.pipelines.train

optimize:
	uv run python -m techchallenge_fase3.pipelines.optimize

benchmark:
	uv run python -m techchallenge_fase3.pipelines.benchmark

api-benchmark:
	uv run python -m techchallenge_fase3.pipelines.api_benchmark

run:
	uv run uvicorn techchallenge_fase3.api.main:app --reload

compose:
	docker compose -f docker/docker-compose.yml up --build

airflow:
	docker compose -f docker/docker-compose.yml --profile airflow up --build
