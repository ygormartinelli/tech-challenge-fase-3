"""Testes do pipeline de modelagem."""

import pandas as pd
import pytest

from techchallenge_fase3.modeling import evaluate_model, train_model


def test_train_model_predicts_known_labels(sample_data: pd.DataFrame) -> None:
    """Treina e avalia um pipeline em dados pequenos."""
    model = train_model(sample_data)
    metrics = evaluate_model(model, sample_data)
    assert list(model.classes_) == [0, 1, 2]
    assert metrics["macro_f1"] > 0.5
    assert len(metrics["confusion_matrix"]) == 3


def test_training_rejects_incomplete_target(sample_data: pd.DataFrame) -> None:
    """Não permite silenciosamente um classificador com classes ausentes."""
    with pytest.raises(ValueError, match="all three"):
        train_model(sample_data[sample_data.urgency_label.ne(2)])
