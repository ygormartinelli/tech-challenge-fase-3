"""Leitura e validação do contrato de dados."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"condition_label", "medical_abstract"}


def load_dataset(path: Path) -> pd.DataFrame:
    """Carrega e valida um CSV de resumos médicos.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        Dados validados.
    """
    dataset = pd.read_csv(path)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Valida as colunas e os valores essenciais do dataset."""
    missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if dataset.empty or dataset["medical_abstract"].isna().any():
        raise ValueError("Dataset must contain non-null medical abstracts")
    if dataset["condition_label"].isna().any():
        raise ValueError("Dataset must contain non-null labels")


def load_label_names(path: Path) -> dict[int, str]:
    """Carrega o mapeamento de rótulos para nomes de categorias."""
    labels = pd.read_csv(path)
    return dict(zip(labels["condition_label"], labels["condition_name"], strict=True))
