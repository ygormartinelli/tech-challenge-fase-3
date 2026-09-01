"""Benchmark reproduzível da inferência original e otimizada."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from techchallenge_fase3.artifacts import OnnxPredictor, load_original, optimized_path
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_dataset
from techchallenge_fase3.modeling import predictions_match

ITERATIONS = 200
WARMUP_ITERATIONS = 20


def measure_latency(predictor: Any, text: str) -> dict[str, float]:
    """Mede latências de inferência individual após aquecimento."""
    for _ in range(WARMUP_ITERATIONS):
        predictor.predict([text])
    durations = [_prediction_time(predictor, text) for _ in range(ITERATIONS)]
    values = np.asarray(durations)
    return {
        "mean_ms": float(values.mean() * 1_000),
        "p50_ms": float(np.percentile(values, 50) * 1_000),
        "p95_ms": float(np.percentile(values, 95) * 1_000),
        "throughput_per_second": float(1 / values.mean()),
    }


def _prediction_time(predictor: Any, text: str) -> float:
    """Mede uma única predição em segundos."""
    start = perf_counter()
    predictor.predict([text])
    return perf_counter() - start


def main() -> None:
    """Compara modelos e registra o resultado do benchmark em JSON."""
    settings = Settings()
    test_data = load_dataset(settings.test_path)
    texts = test_data["medical_abstract"].head(50).tolist()
    original = load_original(settings.model_dir)
    optimized = OnnxPredictor(optimized_path(settings.model_dir))
    equivalent = predictions_match(original.predict(texts), optimized.predict(texts))
    report = _build_report(original, optimized, texts[0], equivalent)
    _write_report(report, Path("reports/latency_benchmark.json"))
    if not equivalent or report["optimized"]["p50_ms"] >= report["original"]["p50_ms"]:
        raise SystemExit(
            "Optimized model did not meet the latency acceptance criterion"
        )


def _build_report(
    original: Any, optimized: Any, text: str, equivalent: bool
) -> dict[str, Any]:
    """Monta o relatório comparativo de latência."""
    return {
        "prediction_equivalent": equivalent,
        "original": measure_latency(original, text),
        "optimized": measure_latency(optimized, text),
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    """Persiste o relatório JSON do benchmark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
