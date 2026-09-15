"""Leitura e validação do contrato de dados."""

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"condition_label", "medical_abstract"}
LABEL_NAMES = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}


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
    texts = dataset["medical_abstract"]
    if not texts.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Medical abstracts must be non-blank strings")
    labels = dataset["condition_label"]
    if not pd.api.types.is_integer_dtype(labels) or not labels.isin(LABEL_NAMES).all():
        raise ValueError("Labels must be integers from 1 to 5")


def load_label_names(path: Path) -> dict[int, str]:
    """Carrega o mapeamento de rótulos para nomes de categorias."""
    labels = pd.read_csv(path)
    mapping = dict(
        zip(labels["condition_label"], labels["condition_name"], strict=True)
    )
    if len(labels) != 5 or mapping != LABEL_NAMES:
        raise ValueError("Label catalogue does not match the five supported categories")
    return mapping


def text_key(text: str) -> str:
    """Identifica textos equivalentes sem expor o conteúdo nos relatórios."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def text_groups(dataset: pd.DataFrame) -> pd.Series:
    """Agrupa abstracts por Unicode, caixa e espaços normalizados."""
    return dataset["medical_abstract"].map(text_key)
