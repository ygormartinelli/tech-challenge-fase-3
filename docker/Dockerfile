FROM ghcr.io/astral-sh/uv:0.11.8 AS uv
FROM python:3.11-slim-bookworm

WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install --yes --no-install-recommends locales \
    && sed --in-place '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm --recursive --force /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --group build --no-install-project --python /usr/local/bin/python \
    && uv sync --frozen --no-dev --group build --no-build-isolation --no-editable --python /usr/local/bin/python \
    && uv pip check --python /app/.venv/bin/python

USER app
EXPOSE 8000
CMD ["uvicorn", "techchallenge_fase3.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
