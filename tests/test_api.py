"""Testes da interface HTTP."""

import numpy as np
from fastapi.testclient import TestClient

from techchallenge_fase3.api.main import create_app


class FakePredictor:
    """Preditor determinístico usado pela suíte de testes."""

    def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Retorna classes e probabilidades em uma única passagem."""
        return np.asarray([1] * len(texts)), np.asarray(
            [[0.9, 0.025, 0.025, 0.025, 0.025]] * len(texts)
        )


def test_predict_returns_category_and_disclaimer() -> None:
    """Expõe a categoria real e a advertência de segurança."""
    app = create_app()
    app.state.service = (FakePredictor(), "optimized")
    payload = {"medical_abstract": "A long enough medical abstract."}
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["condition_name"] == "neoplasms"
    assert "not for clinical" in response.json()["disclaimer"]


def test_metrics_and_validation_are_exposed() -> None:
    """Expõe métricas e rejeita textos muito curtos."""
    app = create_app()
    client = TestClient(app)
    assert client.get("/metrics").status_code == 200
    response = client.post("/predict", json={"medical_abstract": "short"})
    assert response.status_code == 422
