"""Testes dos artefatos de inferência."""

from pathlib import Path

import pandas as pd

from techchallenge_fase3.artifacts import OnnxPredictor, export_onnx, save_original
from techchallenge_fase3.modeling import predictions_match, train_model


def test_onnx_predictions_match_original(tmp_path: Path) -> None:
    """Preserva os rótulos ao converter um pipeline para ONNX."""
    dataset = pd.DataFrame(
        {
            "condition_label": [1, 1, 2, 2],
            "medical_abstract": [
                "tumor cell",
                "cancer cell",
                "colon bowel",
                "liver bowel",
            ],
        }
    )
    model = train_model(dataset)
    save_original(model, tmp_path)
    onnx_path = export_onnx(model, tmp_path)
    texts = ["tumor cancer", "bowel colon"]
    onnx_predictions = OnnxPredictor(onnx_path).predict(texts)
    assert predictions_match(model.predict(texts), onnx_predictions)
