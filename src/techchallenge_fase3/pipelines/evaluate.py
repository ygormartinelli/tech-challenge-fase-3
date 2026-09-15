"""Avaliação final, sem seleção de hiperparâmetros pelo teste."""

from techchallenge_fase3.artifacts import load_original
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_dataset
from techchallenge_fase3.modeling import evaluate_model, test_partitions
from techchallenge_fase3.reporting import environment, file_hash, write_json


def main() -> None:
    """Reporta teste fornecido, textos vistos e textos inéditos separadamente."""
    settings = Settings()
    train, test = load_dataset(settings.train_path), load_dataset(settings.test_path)
    model = load_original(settings.candidate_dir)
    report = {
        name: evaluate_model(model, data) if len(data) else None
        for name, data in test_partitions(train, test).items()
    }
    report["environment"] = environment()
    report["test_sha256"] = file_hash(settings.test_path)
    report["model_sha256"] = file_hash(settings.candidate_dir / "classifier.joblib")
    write_json(settings.report_dir / "evaluation.json", report)
    print("Evaluation saved, with explicit seen/unseen subsets.")


if __name__ == "__main__":
    main()
