"""Medição reproduzível de latência fim a fim da API local."""

import json
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

API_URL = "http://localhost:8000/predict"
ITERATIONS = 100
WARMUP_ITERATIONS = 10
SAMPLE_TEXT = "Tumor cells were investigated in the patient and cancer treatment was discussed."


def request_prediction(url: str, text: str) -> None:
    """Envia uma predição HTTP sem persistir dados clínicos."""
    payload = json.dumps({"medical_abstract": text}).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10):
        pass


def measure_api_latency(url: str, text: str) -> dict[str, float]:
    """Mede a latência HTTP após aquecimento da API."""
    for _ in range(WARMUP_ITERATIONS):
        request_prediction(url, text)
    durations = [_request_duration(url, text) for _ in range(ITERATIONS)]
    return summarize_durations(durations)


def _request_duration(url: str, text: str) -> float:
    """Mede uma requisição HTTP em segundos."""
    started_at = perf_counter()
    request_prediction(url, text)
    return perf_counter() - started_at


def summarize_durations(durations: list[float]) -> dict[str, float]:
    """Consolida duração média, percentis e throughput."""
    values = np.asarray(durations)
    return {
        "mean_ms": float(values.mean() * 1_000),
        "p50_ms": float(np.percentile(values, 50) * 1_000),
        "p95_ms": float(np.percentile(values, 95) * 1_000),
        "throughput_per_second": float(1 / values.mean()),
    }


def main() -> None:
    """Imprime o relatório de latência HTTP em JSON."""
    print(json.dumps(measure_api_latency(API_URL, SAMPLE_TEXT), indent=2))


if __name__ == "__main__":
    main()
