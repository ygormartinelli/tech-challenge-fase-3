"""Testes dos artefatos de inferência."""

from pathlib import Path

import numpy as np
import pandas as pd

from techchallenge_fase3.artifacts import OnnxPredictor, export_onnx, save_original
from techchallenge_fase3.modeling import predictions_match, train_model


def test_onnx_predictions_match_original(
    tmp_path: Path, sample_data: pd.DataFrame
) -> None:
    """Preserva os rótulos ao converter um pipeline para ONNX."""
    model = train_model(sample_data)
    save_original(model, tmp_path)
    onnx_path = export_onnx(model, tmp_path)
    texts = sample_data.report_text.tolist() + ["EXAME SEM ALTERAÇÕES agudas."]
    labels, probabilities = OnnxPredictor(onnx_path).predict_batch(texts)
    assert predictions_match(model.predict(texts), labels)
    np.testing.assert_allclose(model.predict_proba(texts), probabilities, atol=1e-5)
