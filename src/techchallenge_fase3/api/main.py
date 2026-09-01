"""API FastAPI para classificação de resumos médicos."""

from time import perf_counter
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from techchallenge_fase3.artifacts import load_predictor
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_label_names

REQUESTS = Counter(
    "medical_classifier_requests_total",
    "Total de requisições HTTP recebidas.",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "medical_classifier_request_duration_seconds",
    "Duração das requisições HTTP em segundos.",
    ["method", "path"],
)
ERRORS = Counter(
    "medical_classifier_prediction_errors_total",
    "Total de falhas durante inferências.",
)
DISCLAIMER = "Educational use only; not for clinical diagnosis, triage, or treatment."


class PredictionRequest(BaseModel):
    """Payload de classificação."""

    medical_abstract: str = Field(min_length=10, max_length=20_000)


class PredictionResponse(BaseModel):
    """Resposta segura da classificação."""

    condition_label: int
    condition_name: str
    confidence: float
    model_variant: str
    disclaimer: str = DISCLAIMER


def create_app() -> FastAPI:
    """Cria a aplicação FastAPI instrumentada."""
    application = FastAPI(title="Medical Text Category Classifier", version="0.1.0")
    _add_routes(application)
    return application


def _add_routes(application: FastAPI) -> None:
    """Registra as rotas públicas da API."""
    application.get("/health")(_health)
    application.get("/metrics")(_metrics)
    application.post("/predict", response_model=PredictionResponse)(_predict)
    application.middleware("http")(_record_metrics)


async def _record_metrics(request: Request, call_next: Any) -> Response:
    """Registra contagem, estado e duração de cada requisição."""
    started_at = perf_counter()
    response = await call_next(request)
    path = request.url.path
    LATENCY.labels(request.method, path).observe(perf_counter() - started_at)
    REQUESTS.labels(request.method, path, response.status_code).inc()
    return response


def _health(request: Request) -> dict[str, str]:
    """Informa se o artefato de modelo está disponível."""
    _get_service(request)
    return {"status": "ok"}


def _metrics() -> Response:
    """Expõe métricas no formato Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Classifica um resumo médico sem alegar decisão clínica."""
    try:
        predictor, labels, variant = _get_service(request)
        label = int(predictor.predict([payload.medical_abstract])[0])
        probabilities = predictor.predict_proba([payload.medical_abstract])[0]
        confidence = float(np.max(probabilities))
        return PredictionResponse(
            condition_label=label,
            condition_name=labels[label],
            confidence=confidence,
            model_variant=variant,
        )
    except (KeyError, OSError, ValueError) as error:
        ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model unavailable") from error


def _get_service(request: Request) -> tuple[Any, dict[int, str], str]:
    """Carrega o modelo e o catálogo uma vez por processo."""
    if not hasattr(request.app.state, "service"):
        settings = Settings()
        predictor, variant = load_predictor(settings.model_dir, settings.model_variant)
        request.app.state.service = (
            predictor,
            load_label_names(settings.labels_path),
            variant,
        )
    return request.app.state.service


app = create_app()
