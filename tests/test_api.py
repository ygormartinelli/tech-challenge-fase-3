"""Testes da interface HTTP."""

import numpy as np
from fastapi.testclient import TestClient

from techchallenge_fase3.api.main import create_app


class FakePredictor:
    """Preditor determinístico usado pela suíte de testes."""

    def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Retorna classes e probabilidades em uma única passagem."""
        return np.asarray([2] * len(texts)), np.asarray(
            [[0.05, 0.05, 0.9]] * len(texts)
        )


def test_predict_returns_urgency_and_disclaimer() -> None:
    """Expõe urgência simulada, proveniência e advertência de segurança."""
    app = create_app()
    app.state.service = (FakePredictor(), "optimized")
    payload = {"report_text": "A long enough medical abstract."}
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["urgency"] == "urgente"
    assert response.json()["urgency_label"] == 2
    assert response.json()["confidence"] == 0.9
    assert response.json()["data_origin"] == "synthetic"
    assert response.json()["requires_human_review"] is True
    assert "Não utilizar para triagem clínica" in response.json()["disclaimer"]


def test_metrics_and_validation_are_exposed() -> None:
    """Expõe métricas e rejeita textos muito curtos."""
    app = create_app()
    client = TestClient(app)
    assert client.get("/metrics").status_code == 200
    response = client.post("/predict", json={"report_text": "short"})
    assert response.status_code == 422
