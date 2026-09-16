"""Aceitação precisa falhar por divergência ou ausência de ganho real."""

from pathlib import Path

import pandas as pd

from techchallenge_fase3.artifacts import OnnxPredictor, OriginalPredictor, export_onnx
from techchallenge_fase3.modeling import train_model
from techchallenge_fase3.pipelines.benchmark import (
    EDGE_TEXTS,
    benchmark_accepted,
    equivalence,
)


def test_probabilities_and_tokenization_match(
    tmp_path: Path, sample_data: pd.DataFrame
) -> None:
    """Testa dígitos, pontuação, caixa e Unicode, além das três urgências."""
    model = train_model(sample_data)
    optimized = OnnxPredictor(export_onnx(model, tmp_path))
    report = equivalence(
        OriginalPredictor(model),
        optimized,
        sample_data.report_text.tolist() + EDGE_TEXTS,
    )
    assert report["passed"]


def test_gate_requires_every_round_and_equivalence() -> None:
    """Não aprova resultado fortuito de uma única rodada."""
    report = {
        "equivalence": {"passed": True},
        "latency": {
            "rounds": [{"original": {"p50_ms": 2}, "optimized": {"p50_ms": 1}}] * 3
        },
    }
    assert benchmark_accepted(report)
    report["latency"]["rounds"].append(
        {"original": {"p50_ms": 1}, "optimized": {"p50_ms": 2}}
    )
    assert not benchmark_accepted(report)
    report["latency"]["rounds"].pop()
    report["equivalence"]["passed"] = False
    assert not benchmark_accepted(report)
    report["equivalence"]["passed"] = True
    report["latency"]["rounds"] = []
    assert not benchmark_accepted(report)
