"""Retreino local com comandos isolados das dependências do Airflow."""

from datetime import UTC, datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

STAGES = ("validate", "train", "optimize", "evaluate", "benchmark", "publish")
PYTHON = "/opt/project/.venv/bin/python"

with DAG(
    dag_id="retrain_medical_classifier",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["synthetic", "urgency", "training"],
    default_args={"retries": 0},
    description="Urgência sintética: validação, retreino e publicação com gate ONNX.",
) as dag:
    previous = None
    for stage in STAGES:
        current = BashOperator(
            task_id=stage,
            bash_command=f"{PYTHON} -m techchallenge_fase3.pipelines.{stage}",
            cwd="/opt/project",
            do_xcom_push=False,
        )
        if previous is not None:
            previous >> current
        previous = current
