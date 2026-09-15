"""Contratos de Compose e DAG sem dependência de scheduler na CI normal."""

import ast
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("validate", "train", "optimize", "evaluate", "benchmark", "publish")


def test_dag_declares_ordered_reusable_commands() -> None:
    """Confere parsing e contrato; importação real é validada no container."""
    tree = ast.parse(
        (ROOT / "dags/retrain_medical_classifier.py").read_text(encoding="utf-8")
    )
    stages = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "STAGES"
            for target in node.targets
        )
    )
    assert ast.literal_eval(stages) == STAGES
    for stage in STAGES:
        assert (ROOT / f"src/techchallenge_fase3/pipelines/{stage}.py").exists()
    source = ast.unparse(tree)
    assert "max_active_runs=1" in source
    assert "previous >> current" in source
    assert "do_xcom_push=False" in source


def test_compose_has_optional_airflow_and_readonly_api_models() -> None:
    """Impede instalação obrigatória do Airflow e escrita pela API."""
    compose = yaml.safe_load((ROOT / "docker/docker-compose.yml").read_text())
    services = compose["services"]
    assert services["airflow"]["profiles"] == ["airflow"]
    assert "../models:/app/models:ro" in services["api"]["volumes"]
    assert services["api"]["healthcheck"]
    assert set(services) == {"api", "prometheus", "grafana", "airflow"}


def test_grafana_datasource_matches_all_panels() -> None:
    """Regressão do UID ausente encontrado durante a revisão."""
    folder = ROOT / "docker/grafana/provisioning"
    source = yaml.safe_load((folder / "datasources/prometheus.yml").read_text())
    dashboard = json.loads(
        (folder / "dashboards/medical-classifier.json").read_text(encoding="utf-8")
    )
    assert source["datasources"][0]["uid"] == "prometheus"
    assert len(dashboard["panels"]) >= 3
    assert all(
        panel["datasource"]["uid"] == "prometheus" for panel in dashboard["panels"]
    )
    assert any(
        "rate(" in target["expr"] and "errors_total" in target["expr"]
        for panel in dashboard["panels"]
        for target in panel["targets"]
    )
