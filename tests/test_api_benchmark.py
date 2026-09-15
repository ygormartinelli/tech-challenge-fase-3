"""Testes da consolidação de latência da API."""

import pytest

from techchallenge_fase3.pipelines.api_benchmark import summarize_durations


def test_summarize_durations_calculates_expected_values() -> None:
    """Consolida estatísticas de duração em milissegundos."""
    summary = summarize_durations([0.001, 0.002, 0.003])
    assert summary["mean_ms"] == 2.0
    assert summary["p50_ms"] == 2.0
    assert summary["throughput_per_second"] == 500.0


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_durations_are_rejected(values: list[float]) -> None:
    """Impede métricas vazias, infinitas ou durações impossíveis."""
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_durations(values)
