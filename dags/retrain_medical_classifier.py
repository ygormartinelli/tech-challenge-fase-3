"""DAG de retreino do classificador de categorias médicas."""

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["medical", "training"],
)
def retrain_medical_classifier() -> None:
    """Orquestra a validação, treino, otimização e benchmark."""

    @task
    def validate_data() -> None:
        """Valida o contrato do arquivo de treino."""
        from techchallenge_fase3.config import Settings
        from techchallenge_fase3.data import load_dataset

        load_dataset(Settings().train_path)

    @task
    def train() -> None:
        """Executa o pipeline reutilizável de treino."""
        from techchallenge_fase3.pipelines.train import main

        main()

    @task
    def optimize() -> None:
        """Executa a exportação ONNX reutilizável."""
        from techchallenge_fase3.pipelines.optimize import main

        main()

    @task
    def benchmark() -> None:
        """Executa o benchmark e critério de aceitação."""
        from techchallenge_fase3.pipelines.benchmark import main

        main()

    validate_data() >> train() >> optimize() >> benchmark()


retrain_medical_classifier()
