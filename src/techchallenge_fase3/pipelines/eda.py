"""Executa e preserva a EDA agregada dos CSVs locais."""

from techchallenge_fase3.analysis import describe_corpus
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_dataset, load_label_names
from techchallenge_fase3.reporting import environment, file_hash, write_json


def main() -> None:
    """Gera evidência reproduzível sem exportar registros médicos."""
    settings = Settings()
    load_label_names(settings.labels_path)
    report = describe_corpus(
        load_dataset(settings.train_path), load_dataset(settings.test_path)
    )
    report["environment"] = environment()
    report["sources"] = {
        path.name: file_hash(path)
        for path in (settings.train_path, settings.test_path, settings.labels_path)
    }
    write_json(settings.report_dir / "eda.json", report)
    print(f"EDA saved: {settings.report_dir / 'eda.json'}")


if __name__ == "__main__":
    main()
