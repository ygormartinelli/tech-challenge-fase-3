"""Validação de entrada, readiness e métricas em cenários de falha."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from techchallenge_fase3.api.main import create_app


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"medical_abstract": "Old incompatible input field"},
        {"report_text": " " * 20},
        {"report_text": 12},
        {"report_text": "a" * 20001},
        {"report_text": "valid long text", "urgency": "urgent"},
    ],
)
def test_malformed_payloads(payload: dict) -> None:
    """Rejeita entradas fora do contrato antes de acessar o modelo."""
    assert TestClient(create_app()).post("/predict", json=payload).status_code == 422


def test_unavailable_models_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Não vaza a exceção interna nem o texto recebido."""

    def unavailable(*args: object) -> None:
        raise RuntimeError("private path and sensitive text")

    monkeypatch.setattr("techchallenge_fase3.api.main.load_predictor", unavailable)
    client = TestClient(create_app())
    assert client.get("/health").status_code == 503
    response = client.post(
        "/predict", json={"report_text": "sensitive medical abstract"}
    )
    assert response.json() == {"detail": "Model unavailable"}
    metrics = client.get("/metrics").text
    assert "medical_classifier_prediction_errors_total 1.0" in metrics
    assert 'path="/predict",status="503"' in metrics
    assert "sensitive" not in metrics


def test_metrics_limit_cardinality_and_exclude_scrapes() -> None:
    """URLs arbitrárias não criam séries infinitas; scraping não conta tráfego."""
    client = TestClient(create_app())
    client.get("/unknown-one")
    client.get("/unknown-two")
    metrics = client.get("/metrics").text
    assert 'path="unmatched",status="404"} 2.0' in metrics
    assert 'path="/metrics"' not in metrics
    assert "unknown-one" not in metrics


@pytest.mark.parametrize(
    "probabilities",
    [[[0.2] * 5], [[float("nan"), 0.5, 0.5]], [[-0.1, 0.5, 0.6]], [[0.1] * 3]],
)
def test_invalid_model_outputs_fail_closed(probabilities: list[list[float]]) -> None:
    """Saídas antigas ou inválidas não viram respostas de urgência plausíveis."""

    class BrokenPredictor:
        def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
            return np.asarray([0]), np.asarray(probabilities)

    app = create_app()
    app.state.service = (BrokenPredictor(), "optimized")
    response = TestClient(app).post("/predict", json={"report_text": "Exame fictício."})
    assert response.status_code == 503
    assert response.json() == {"detail": "Model unavailable"}
