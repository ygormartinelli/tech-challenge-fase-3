"""Latência HTTP reproduzível com textos variados e respostas completas."""

import json
import os
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np
from dotenv import load_dotenv

from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import LABEL_NAMES, TASK_ID, load_dataset
from techchallenge_fase3.reporting import environment, file_hash, write_json

ITERATIONS = 200
WARMUP_ITERATIONS = 20


def request_prediction(url: str, text: str) -> dict:
    """Recebe e valida a resposta completa, sem persistir abstracts."""
    payload = json.dumps({"report_text": text}).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
        if result.get("urgency_label") not in LABEL_NAMES:
            raise ValueError("API returned an invalid prediction")
        if result.get("data_origin") != "synthetic":
            raise ValueError("API did not declare the synthetic origin")
        return result


def summarize_durations(durations: list[float]) -> dict[str, float]:
    """Consolida latência; throughput é sequencial, não teste de saturação."""
    values = np.asarray(durations)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Durations must be finite and positive")
    return {
        "mean_ms": float(values.mean() * 1_000),
        "p50_ms": float(np.percentile(values, 50) * 1_000),
        "p95_ms": float(np.percentile(values, 95) * 1_000),
        "throughput_per_second": float(1 / values.mean()),
    }


def measure_api_latency(url: str, texts: list[str], variant: str) -> dict[str, float]:
    """Mantém o corpus e valida a variante em todas as chamadas medidas."""
    for text in texts[:WARMUP_ITERATIONS]:
        request_prediction(url, text)
    durations = []
    for text in texts:
        start = perf_counter()
        result = request_prediction(url, text)
        durations.append(perf_counter() - start)
        if result["model_variant"] != variant:
            raise ValueError("Model variant changed during benchmark")
    return summarize_durations(durations)


def main() -> None:
    """Grava um relatório por variante que efetivamente está em execução."""
    load_dotenv()
    settings = Settings()
    base = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"
    with urlopen(base + "/health", timeout=10) as response:
        variant = json.load(response)["model_variant"]
    data = load_dataset(settings.test_path)
    texts = data.sample(n=ITERATIONS, random_state=42).report_text.tolist()
    report = {
        "task_id": TASK_ID,
        "data_origin": "synthetic",
        "environment": environment(),
        "variant": variant,
        "seed": 42,
        "iterations": ITERATIONS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "test_sha256": file_hash(settings.test_path),
        "latency": measure_api_latency(base + "/predict", texts, variant),
    }
    write_json(settings.report_dir / f"http_{variant}.json", report)
    print(json.dumps(report["latency"], indent=2))


if __name__ == "__main__":
    main()
