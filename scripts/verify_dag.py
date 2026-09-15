"""Importação real da DAG; executar com o Python da imagem Airflow."""

from airflow.models import DagBag

EXPECTED = ["validate", "train", "optimize", "evaluate", "benchmark", "publish"]


def main() -> None:
    """Valida a cadeia completa sem exigir scheduler ou execução de treino."""
    bag = DagBag(dag_folder="/opt/airflow/dags", include_examples=False)
    assert not bag.import_errors, bag.import_errors
    dag = bag.dags.get("retrain_medical_classifier")
    assert dag is not None
    assert [task.task_id for task in dag.topological_sort()] == EXPECTED
    assert dag.max_active_runs == 1
    for index, name in enumerate(EXPECTED):
        task = dag.get_task(name)
        assert task.bash_command.endswith(f"techchallenge_fase3.pipelines.{name}")
        assert task.upstream_task_ids == ({EXPECTED[index - 1]} if index else set())
    print("DAG import and six task contracts verified in real Airflow.")


if __name__ == "__main__":
    main()
