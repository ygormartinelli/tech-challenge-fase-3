"""Contrato de entrada compartilhado entre CLI e Airflow."""

from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import LABEL_NAMES, load_dataset, load_label_names


def main() -> None:
    """Exige catálogo consistente, cinco classes e volume mínimo de treino."""
    settings = Settings()
    load_label_names(settings.labels_path)
    for path in (settings.train_path, settings.test_path):
        data = load_dataset(path)
        if set(data.condition_label) != set(LABEL_NAMES):
            raise ValueError("All five categories must occur in each split")
        if path == settings.train_path and len(data) < 2000:
            raise ValueError("Training requires at least 2000 rows")
    print("Raw data contract validated.")


if __name__ == "__main__":
    main()
