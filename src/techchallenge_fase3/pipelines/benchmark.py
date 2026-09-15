"""Benchmark pareado de requisições individuais e gate de equivalência."""

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnx

from techchallenge_fase3.artifacts import (
    OPTIMIZED_MODEL,
    ORIGINAL_MODEL,
    OnnxPredictor,
    OriginalPredictor,
    Predictor,
    load_original,
    optimized_path,
)
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_dataset
from techchallenge_fase3.pipelines.api_benchmark import summarize_durations
from techchallenge_fase3.reporting import environment, file_hash, write_json

ITERATIONS = 200
WARMUP_ITERATIONS = 20
ROUNDS = 3
PROBABILITY_TOLERANCE = 1e-5
EDGE_TEXTS = [
    "Tumor cells, cancer: treatment! 123.",
    "CARDIOVASCULAR disease\nheart\tpressure",
    "café naïve β receptor and tumor cells",
    "unknownwordzzzz",
    "  bowel    digestive  disease  ",
    "delta f508 mutations; tumor tumor tumor; x 1 y heart",
]


def equivalence(
    first: Predictor, second: Predictor, texts: list[str]
) -> dict[str, Any]:
    """Confere todos os rótulos e probabilidades, em lotes limitados."""
    mismatches, maximum_error = 0, 0.0
    for start in range(0, len(texts), 128):
        batch = texts[start : start + 128]
        labels_a, probabilities_a = first.predict_batch(batch)
        labels_b, probabilities_b = second.predict_batch(batch)
        mismatches += int(np.count_nonzero(labels_a != labels_b))
        maximum_error = max(
            maximum_error, float(np.abs(probabilities_a - probabilities_b).max())
        )
    return {
        "rows": len(texts),
        "label_mismatches": mismatches,
        "max_probability_absolute_error": maximum_error,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "passed": mismatches == 0 and maximum_error <= PROBABILITY_TOLERANCE,
    }


def timed_predictions(predictor: Predictor, texts: list[str]) -> list[float]:
    """Mede o mesmo contrato usado pela API: rótulos e probabilidades."""
    durations = []
    for text in texts:
        start = perf_counter()
        predictor.predict_batch([text])
        durations.append(perf_counter() - start)
    return durations


def compare_latency(
    first: Predictor, second: Predictor, texts: list[str]
) -> dict[str, Any]:
    """Alterna a ordem dos modelos em três rodadas do mesmo corpus fixo."""
    predictors = {"original": first, "optimized": second}
    for predictor in predictors.values():
        timed_predictions(predictor, texts[:WARMUP_ITERATIONS])
    samples: dict[str, list[float]] = {name: [] for name in predictors}
    rounds = []
    for index in range(ROUNDS):
        names = list(predictors) if index % 2 == 0 else list(reversed(predictors))
        measured = {name: timed_predictions(predictors[name], texts) for name in names}
        rounds.append(
            {name: summarize_durations(values) for name, values in measured.items()}
        )
        for name, values in measured.items():
            samples[name].extend(values)
    return {
        **{name: summarize_durations(values) for name, values in samples.items()},
        "rounds": rounds,
    }


def benchmark_accepted(report: dict[str, Any]) -> bool:
    """Exige paridade e melhoria real em cada rodada, sem limiar artificial."""
    rounds = report["latency"]["rounds"]
    return (
        len(rounds) == ROUNDS
        and report["equivalence"]["passed"]
        and all(
            row["optimized"]["p50_ms"] < row["original"]["p50_ms"] for row in rounds
        )
    )


def quantization_assessment(path: Path) -> dict[str, Any]:
    """Registra se a quantização dinâmica padrão teria operadores elegíveis."""
    operators = sorted({node.op_type for node in onnx.load(str(path)).graph.node})
    eligible = set(operators) & {"MatMul", "Gemm", "LSTM"}
    return {
        "graph_operators": operators,
        "eligible_dynamic_operators": sorted(eligible),
        "applied": False,
        "reason": "Mantido ONNX float; não há ganho de quantização demonstrado.",
    }


def build_report(settings: Settings) -> dict[str, Any]:
    """Usa o teste completo para paridade e amostra fixa para latência."""
    dataset = load_dataset(settings.test_path)
    texts = dataset.medical_abstract.tolist()
    rng = np.random.default_rng(42)
    sample = [
        texts[index]
        for index in rng.choice(len(texts), ITERATIONS, replace=len(texts) < ITERATIONS)
    ]
    first = OriginalPredictor(load_original(settings.candidate_dir))
    second = OnnxPredictor(optimized_path(settings.candidate_dir))
    return {
        "environment": environment(),
        "seed": 42,
        "iterations_per_round": ITERATIONS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "rounds": ROUNDS,
        "contract": "single text -> label and probabilities; sequential, CPU",
        "test_sha256": file_hash(settings.test_path),
        "model_hashes": {
            name: file_hash(settings.candidate_dir / name)
            for name in (ORIGINAL_MODEL, OPTIMIZED_MODEL)
        },
        "equivalence": equivalence(first, second, texts + EDGE_TEXTS),
        "latency": compare_latency(first, second, sample),
        "quantization": quantization_assessment(optimized_path(settings.candidate_dir)),
    }


def main() -> None:
    """Salva o resultado mesmo quando o gate rejeita o modelo otimizado."""
    settings = Settings()
    report = build_report(settings)
    report["optimized_approved"] = benchmark_accepted(report)
    write_json(settings.report_dir / "latency_benchmark.json", report)
    if not report["optimized_approved"]:
        raise SystemExit("Optimization rejected: equivalence or latency failed")
    print("Optimization accepted in all rounds; report saved.")


if __name__ == "__main__":
    main()
