"""Registra evidência de uma execução real da DAG, sem expor credenciais."""

import json
import subprocess
import sys
from pathlib import Path

from techchallenge_fase3.reporting import environment, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Exige sucesso nas seis tarefas e no benchmark produzido pelo Airflow."""
    command = [
        "docker",
        "compose",
        "-f",
        "docker/docker-compose.yml",
        "exec",
        "-T",
        "airflow",
        "airflow",
        "tasks",
        "states-for-dag-run",
        "retrain_medical_classifier",
        sys.argv[1],
        "--output",
        "json",
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, cwd=ROOT
    )
    tasks = json.loads(result.stdout)
    assert len(tasks) == 6 and all(task["state"] == "success" for task in tasks)
    benchmark = json.loads(
        (ROOT / "reports/runtime/airflow/latency_benchmark.json").read_text()
    )
    assert benchmark["optimized_approved"]
    report = {
        "environment": environment(),
        "run_id": sys.argv[1],
        "tasks": tasks,
        "runtime_benchmark": benchmark,
        "scheduler_executed": True,
    }
    write_json(ROOT / "reports/airflow_verification.json", report)
    print("Six real Airflow tasks completed successfully; evidence saved.")


if __name__ == "__main__":
    main()
