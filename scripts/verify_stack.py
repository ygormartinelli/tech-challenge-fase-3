"""Smoke test da API, coleta Prometheus e consultas reais do Grafana."""

import base64
import json
import math
import os
from pathlib import Path
from time import sleep
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from techchallenge_fase3.reporting import environment, write_json

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"
PROMETHEUS = f"http://127.0.0.1:{os.getenv('PROMETHEUS_PORT', '9090')}"
GRAFANA = f"http://127.0.0.1:{os.getenv('GRAFANA_PORT', '3000')}"


def read_json(
    url: str, headers: dict | None = None, payload: dict | None = None
) -> dict:
    """Lê a resposta completa, com timeout e sem registrar credenciais."""
    request = Request(
        url,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def grafana_headers() -> dict[str, str]:
    """Usa apenas as credenciais locais explicitadas na configuração."""
    password = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")
    credentials = base64.b64encode(f"admin:{password}".encode()).decode()
    return {"Authorization": f"Basic {credentials}"}


def generate_traffic() -> None:
    """Gera sucessos e 422 durante três coletas para habilitar taxas."""
    payload = {"medical_abstract": "Tumor cells were investigated in cancer treatment."}
    for _ in range(4):
        for _ in range(5):
            result = read_json(API + "/predict", payload=payload)
            assert result["condition_label"] in range(1, 6)
        try:
            read_json(API + "/predict", payload={"medical_abstract": " "})
        except HTTPError as error:
            assert error.code == 422
        else:
            raise AssertionError("Invalid payload was accepted")
        sleep(5)


def verify_queries() -> list[dict]:
    """Executa cada consulta do dashboard através do próprio Grafana."""
    headers = grafana_headers()
    source = read_json(GRAFANA + "/api/datasources/uid/prometheus", headers)
    assert source["url"] == "http://prometheus:9090"
    dashboard = read_json(GRAFANA + "/api/dashboards/uid/medical-classifier", headers)
    queries = []
    for panel in dashboard["dashboard"]["panels"]:
        for target in panel["targets"]:
            query = urlencode({"query": target["expr"]})
            endpoint = "/api/datasources/proxy/uid/prometheus/api/v1/query?"
            result = read_json(GRAFANA + endpoint + query, headers)
            values = [float(row["value"][1]) for row in result["data"]["result"]]
            assert values and all(math.isfinite(value) for value in values), panel[
                "title"
            ]
            queries.append(
                {"panel": panel["title"], "query": target["expr"], "values": values}
            )
    return queries


def main() -> None:
    """Falha quando uma dependência operacional não responde corretamente."""
    health = read_json(API + "/health")
    assert health["status"] == "ok"
    generate_traffic()
    targets = read_json(PROMETHEUS + "/api/v1/targets")["data"]["activeTargets"]
    assert any(
        target["health"] == "up" and "api:8000" in target["scrapeUrl"]
        for target in targets
    )
    report = {
        "environment": environment(),
        "api_health": health,
        "prometheus_target_up": True,
        "grafana_queries": verify_queries(),
    }
    write_json(ROOT / "reports/stack_verification.json", report)
    print("API, Prometheus and all Grafana panel queries verified.")


if __name__ == "__main__":
    main()
