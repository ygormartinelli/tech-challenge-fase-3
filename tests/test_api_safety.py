"""Validação de entrada, readiness e métricas em cenários de falha."""

import pytest
from fastapi.testclient import TestClient

from techchallenge_fase3.api.main import create_app


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"medical_abstract": " " * 20},
        {"medical_abstract": 12},
        {"medical_abstract": "a" * 20001},
        {"medical_abstract": "valid long text", "urgency": "urgent"},
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
        "/predict", json={"medical_abstract": "sensitive medical abstract"}
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
