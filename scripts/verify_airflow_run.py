"""Registra a execução real de um run concluído pelo scheduler local."""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from techchallenge_fase3.data import TASK_ID
from techchallenge_fase3.reporting import environment, write_json

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"validate", "train", "optimize", "evaluate", "benchmark", "publish"}


def verify(run_id: str) -> dict:
    """Exige seis sucessos e relatório da tarefa correta antes de registrar."""
    command = [
        "docker",
        "compose",
        "--env-file",
        ".env",
        "-f",
        "docker/docker-compose.yml",
        "exec",
        "-T",
        "airflow",
        "airflow",
        "tasks",
        "states-for-dag-run",
        "retrain_medical_classifier",
        run_id,
        "--output",
        "json",
    ]
    result = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    tasks = json.loads(result.stdout)
    assert len(tasks) == 6 and {task["task_id"] for task in tasks} == STAGES
    assert all(task["state"] == "success" for task in tasks)
    benchmark = json.loads(
        (ROOT / "reports/runtime/airflow/latency_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert benchmark["task_id"] == TASK_ID and benchmark["optimized_approved"]
    timing = next(task for task in tasks if task["task_id"] == "benchmark")
    measured = datetime.fromisoformat(benchmark["environment"]["created_at"])
    assert datetime.fromisoformat(timing["start_date"]) <= measured
    assert measured <= datetime.fromisoformat(timing["end_date"])
    pointer = json.loads((ROOT / "models/current.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / "models" / pointer["release"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["model_hashes"] == benchmark["model_hashes"]
    return {
        "environment": environment(),
        "task_id": TASK_ID,
        "data_origin": "synthetic",
        "run_id": run_id,
        "tasks": tasks,
        "runtime_benchmark": benchmark,
        "scheduler_executed": True,
        "published_release": pointer["release"],
    }


def main() -> None:
    """Lê um run existente; não dispara nem modifica o scheduler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    report = verify(parser.parse_args().run_id)
    write_json(ROOT / "reports/airflow_verification.json", report)
    print("Six successful scheduler tasks and published synthetic release verified.")


if __name__ == "__main__":
    main()
