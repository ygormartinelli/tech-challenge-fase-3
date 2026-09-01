"""Testes do pipeline de modelagem."""

import pandas as pd

from techchallenge_fase3.modeling import evaluate_model, train_model


def test_train_model_predicts_known_labels() -> None:
    """Treina e avalia um pipeline em dados pequenos."""
    dataset = pd.DataFrame(
        {
            "condition_label": [1, 1, 2, 2],
            "medical_abstract": [
                "cancer tumor treatment",
                "tumor disease oncology",
                "bowel digestive colon",
                "digestive liver disease",
            ],
        }
    )
    model = train_model(dataset)
    metrics = evaluate_model(model, dataset)
    assert set(model.predict(["colon digestive disease"])).issubset({1, 2})
    assert metrics["macro_f1"] >= 0
