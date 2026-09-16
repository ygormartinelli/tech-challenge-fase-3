"""Contrato de entrada compartilhado entre CLI e Airflow."""

from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_dataset, load_label_names, validate_splits


def main() -> None:
    """Exige origem sintética, classes de urgência e partições independentes."""
    settings = Settings()
    load_label_names(settings.labels_path)
    validate_splits(load_dataset(settings.train_path), load_dataset(settings.test_path))
    print("Synthetic urgency contract validated; no shared scenarios or texts.")


if __name__ == "__main__":
    main()
