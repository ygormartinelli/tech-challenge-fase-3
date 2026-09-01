"""Comando para treinar e persistir o modelo original."""

import json
from pathlib import Path

from techchallenge_fase3.artifacts import save_original
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_dataset
from techchallenge_fase3.modeling import evaluate_model, train_model


def main() -> None:
    """Treina o classificador e registra métricas no diretório de relatórios."""
    settings = Settings()
    train_data = load_dataset(settings.train_path)
    test_data = load_dataset(settings.test_path)
    model = train_model(train_data)
    save_original(model, settings.model_dir)
    report_path = _report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(evaluate_model(model, test_data), indent=2))


def _report_path() -> Path:
    """Retorna o nome do relatório de avaliação."""
    return Path("reports/evaluation.json")


if __name__ == "__main__":
    main()
