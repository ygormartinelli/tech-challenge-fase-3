FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends locales \
    && sed --in-place '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm --recursive --force /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "techchallenge_fase3.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
