"""Instrumentação HTTP com registro isolado e rótulos limitados."""

from time import perf_counter
from typing import Any

from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


class Metrics:
    """Agrupa as três métricas exigidas sem estado global compartilhado."""

    def __init__(self) -> None:
        """Usa buckets adequados à inferência de poucos milissegundos."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "medical_classifier_requests_total",
            "Requisições HTTP.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "medical_classifier_request_duration_seconds",
            "Latência HTTP em segundos.",
            ["method", "path"],
            registry=self.registry,
            buckets=(
                0.001,
                0.0025,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2.5,
                5,
            ),
        )
        self.errors = Counter(
            "medical_classifier_prediction_errors_total",
            "Falhas internas de inferência.",
            registry=self.registry,
        )

    async def record(self, request: Request, call_next: Any) -> Response:
        """Exclui scraping, mas conta falhas HTTP e limita rotas desconhecidas."""
        if request.url.path == "/metrics":
            return await call_next(request)
        started, status = perf_counter(), 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            method = request.method if request.method in {"GET", "POST"} else "OTHER"
            self.latency.labels(method, path).observe(perf_counter() - started)
            self.requests.labels(method, path, str(status)).inc()

    def response(self) -> Response:
        """Renderiza as métricas para o Prometheus."""
        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)
