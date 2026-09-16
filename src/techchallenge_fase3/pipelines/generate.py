"""Gera uma simulação auditável, sem derivar urgência de categorias de doenças."""

import random
from itertools import product
from pathlib import Path

import pandas as pd

from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import LABEL_NAMES, TASK_ID, validate_splits
from techchallenge_fase3.reporting import file_hash, write_json
from techchallenge_fase3.synthetic_cases import PREFIXES, SCENARIOS, SUFFIXES

SEED = 42


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa cenários antes de criar variações de estilo, sem consultar o modelo."""
    splits: dict[str, list[dict]] = {"train": [], "test": []}
    rng = random.Random(SEED)
    for theme, classes in SCENARIOS.items():
        for label, findings in classes.items():
            indices = list(range(len(findings)))
            rng.shuffle(indices)
            for position, index in enumerate(indices):
                split = "train" if position < 3 else "test"
                scenario = f"{theme}-{label}-{index}"
                variants = list(product(PREFIXES, SUFFIXES))
                for prefix, suffix in variants[: 60 if split == "train" else 20]:
                    splits[split].append(
                        {
                            "report_text": f"{prefix} {findings[index]} {suffix}",
                            "urgency_label": label,
                            "scenario_id": scenario,
                            "source_type": "synthetic",
                            "source_theme": theme,
                        }
                    )
    return tuple(pd.DataFrame(splits[name]) for name in ("train", "test"))


def generate(settings: Settings) -> dict:
    """Salva CSVs determinísticos em pasta separada e registra sua proveniência."""
    train, test = build_datasets()
    validate_splits(train, test)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for data, path in ((train, settings.train_path), (test, settings.test_path)):
        data.to_csv(path, index=False, lineterminator="\n")
    pd.DataFrame(LABEL_NAMES.items(), columns=["urgency_label", "urgency"]).to_csv(
        settings.labels_path, index=False, lineterminator="\n"
    )
    report = generation_report(settings, train, test)
    write_json(settings.data_dir / "manifest.json", report)
    write_json(settings.report_dir / "data_generation.json", report)
    return report


def generation_report(
    settings: Settings, train: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Documenta o número efetivo de cenários, sem vender variações como casos reais."""
    return {
        "task_id": TASK_ID,
        "source_type": "synthetic",
        "seed": SEED,
        "language": "pt-BR",
        "clinically_validated": False,
        "adaptation": (
            "Corpus sugerido usado como referência temática, não como fonte "
            "de rótulos de urgência ou textos de treino."
        ),
        "reference": "https://github.com/sebischair/Medical-Abstracts-TC-Corpus",
        "label_policy": (
            "Cenários e rótulos fictícios autorais para simulação acadêmica. "
            "Não são decisões clínicas."
        ),
        "split_policy": (
            "Cenários separados antes da expansão. Não há cenário-base "
            "compartilhado entre treino e teste."
        ),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_scenarios": train.scenario_id.nunique(),
        "test_scenarios": test.scenario_id.nunique(),
        "recipe_sha256": file_hash(Path(__file__).parents[1] / "synthetic_cases.py"),
        "generator_sha256": file_hash(Path(__file__)),
        "files": {
            p.name: file_hash(p)
            for p in (settings.train_path, settings.test_path, settings.labels_path)
        },
    }


def main() -> None:
    """Prepara a simulação autorizada sem modificar os CSVs em data/raw."""
    report = generate(Settings())
    print(
        f"Synthetic corpus: {report['train_rows']} train, {report['test_rows']} test."
    )
    print("Not clinical evidence.")


if __name__ == "__main__":
    main()
