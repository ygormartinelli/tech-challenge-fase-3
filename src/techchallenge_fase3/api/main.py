"""API educacional com inferência única, readiness e falhas seguras."""

import logging
from threading import Lock
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from techchallenge_fase3.api.metrics import Metrics
from techchallenge_fase3.artifacts import Predictor, load_predictor
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import LABEL_NAMES, TASK_ID

DISCLAIMER = (
    "Simulação acadêmica treinada com dados sintéticos. "
    "Não utilizar para triagem clínica, diagnóstico ou tratamento de pacientes."
)
LOGGER = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Rejeita campos inesperados, valores não textuais e espaços vazios."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    report_text: str = Field(min_length=10, max_length=20_000)


class PredictionResponse(BaseModel):
    """Confiança numérica do modelo, não uma probabilidade clínica calibrada."""

    urgency_label: int = Field(ge=0, le=2)
    urgency: Literal["normal", "atenção", "urgente"]
    confidence: float = Field(ge=0, le=1)
    model_variant: str
    data_origin: Literal["synthetic"] = "synthetic"
    requires_human_review: Literal[True] = True
    disclaimer: str = DISCLAIMER


def create_app() -> FastAPI:
    """Constrói instâncias isoladas e testáveis da aplicação."""
    application = FastAPI(
        title="Triagem de urgência — simulação acadêmica",
        version="0.2.0",
        description=DISCLAIMER,
    )
    application.state.metrics = Metrics()
    application.state.load_lock = Lock()
    application.get("/health")(_health)
    application.get("/metrics")(_metrics)
    application.post("/predict", response_model=PredictionResponse)(_predict)
    application.middleware("http")(_record_metrics)
    return application


async def _record_metrics(request: Request, call_next: Any) -> Response:
    """Registra também respostas de erro e limita a cardinalidade das rotas."""
    return await request.app.state.metrics.record(request, call_next)


def _health(request: Request) -> dict[str, str]:
    """Retorna 503 se os artefatos não puderem ser carregados."""
    try:
        _, variant = _get_service(request)
        return {
            "status": "ok",
            "model_variant": variant,
            "task_id": TASK_ID,
            "data_origin": "synthetic",
        }
    except Exception as error:
        raise _unavailable(error) from error


def _metrics(request: Request) -> Response:
    """Expõe apenas o registro Prometheus desta instância."""
    return request.app.state.metrics.response()


def _predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Classifica sem registrar abstracts e sem alegar decisão clínica."""
    try:
        predictor, variant = _get_service(request)
        labels, probabilities = predictor.predict_batch([payload.report_text])
        if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid probability output")
        if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("Invalid probability distribution")
        label = int(labels[0])
        return PredictionResponse(
            urgency_label=label,
            urgency=LABEL_NAMES[label],
            confidence=float(probabilities.max()),
            model_variant=variant,
        )
    except Exception as error:
        request.app.state.metrics.errors.inc()
        raise _unavailable(error) from error


def _unavailable(error: Exception) -> HTTPException:
    """Não expõe caminhos, conteúdo clínico ou detalhes internos ao cliente."""
    LOGGER.error("Model unavailable (%s)", type(error).__name__)
    return HTTPException(status_code=503, detail="Model unavailable")


def _get_service(request: Request) -> tuple[Predictor, str]:
    """Carrega uma release por processo; restart adota a próxima publicação."""
    state = request.app.state
    with state.load_lock:
        if not hasattr(state, "service"):
            settings = Settings()
            state.service = load_predictor(settings.model_dir, settings.model_variant)
    return state.service


app = create_app()
