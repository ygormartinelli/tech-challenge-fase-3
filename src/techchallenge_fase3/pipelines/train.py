"""Comando para treinar e persistir o modelo original."""

from techchallenge_fase3.artifacts import save_original
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import TASK_ID, load_dataset
from techchallenge_fase3.modeling import train_model, validate_baseline
from techchallenge_fase3.reporting import environment, file_hash, write_json


def main() -> None:
    """Treina o classificador e registra métricas no diretório de relatórios."""
    settings = Settings()
    train_data = load_dataset(settings.train_path)
    report = validate_baseline(train_data)
    report.update(task_id=TASK_ID, data_origin="synthetic")
    report["environment"] = environment()
    report["train_sha256"] = file_hash(settings.train_path)
    model = train_model(train_data)
    save_original(model, settings.candidate_dir)
    write_json(settings.report_dir / "validation.json", report)
    print("Baseline validated; candidate trained without test-set fitting.")


if __name__ == "__main__":
    main()
