"""Avaliação final, sem seleção de hiperparâmetros pelo teste."""

from techchallenge_fase3.artifacts import load_original
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import TASK_ID, load_dataset, validate_splits
from techchallenge_fase3.modeling import evaluate_model, scenario_evaluation
from techchallenge_fase3.reporting import environment, file_hash, write_json


def main() -> None:
    """Reporta variações e cenários distintos do teste sintético reservado."""
    settings = Settings()
    train, test = load_dataset(settings.train_path), load_dataset(settings.test_path)
    validate_splits(train, test)
    model = load_original(settings.candidate_dir)
    report = {
        "synthetic_test": evaluate_model(model, test),
        "scenario_test": scenario_evaluation(model, test),
    }
    report["environment"] = environment()
    report.update(task_id=TASK_ID, data_origin="synthetic", clinically_validated=False)
    report["test_sha256"] = file_hash(settings.test_path)
    report["model_sha256"] = file_hash(settings.candidate_dir / "classifier.joblib")
    write_json(settings.report_dir / "evaluation.json", report)
    print(
        "Synthetic evaluation saved at row and scenario levels; not clinical evidence."
    )


if __name__ == "__main__":
    main()
