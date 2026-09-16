"""Leitura e validação do contrato de dados."""

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

TASK_ID = "synthetic_urgency_v1"
LABEL_NAMES = {0: "normal", 1: "atenção", 2: "urgente"}
REQUIRED_COLUMNS = {
    "urgency_label",
    "report_text",
    "scenario_id",
    "source_type",
    "source_theme",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Carrega e valida um CSV de laudos sintéticos.

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
    if dataset.empty or dataset[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Synthetic dataset must be non-empty without nulls")
    texts = dataset["report_text"]
    if not texts.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Report texts must be non-blank strings")
    if not texts.str.len().between(10, 20_000).all():
        raise ValueError("Report text must contain 10 to 20000 characters")
    labels = dataset["urgency_label"]
    if not pd.api.types.is_integer_dtype(labels) or not labels.isin(LABEL_NAMES).all():
        raise ValueError("Urgency labels must be integers 0, 1 or 2")
    validate_provenance(dataset)


def validate_provenance(dataset: pd.DataFrame) -> None:
    """Exige origem sintética explícita e famílias sem conflitos de rótulo."""
    for column in ("scenario_id", "source_theme"):
        if (
            not dataset[column]
            .map(lambda v: isinstance(v, str) and bool(v.strip()))
            .all()
        ):
            raise ValueError(f"Invalid non-blank string column: {column}")
    if not dataset.source_type.eq("synthetic").all():
        raise ValueError("This release accepts only explicitly synthetic data")
    if dataset.groupby("scenario_id").urgency_label.nunique().gt(1).any():
        raise ValueError("Conflicting labels in a synthetic scenario")


def validate_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Impede vazamento de cenários/textos e exige as três classes."""
    for data in (train, test):
        validate_dataset(data)
        if set(data.urgency_label) != set(LABEL_NAMES):
            raise ValueError("All three urgency classes must occur in each split")
    if len(train) < 2000:
        raise ValueError("Training requires at least 2000 synthetic rows")
    if set(train.scenario_id) & set(test.scenario_id):
        raise ValueError("Train/test scenario leakage")
    if set(text_groups(train)) & set(text_groups(test)):
        raise ValueError("Train/test text leakage")


def load_label_names(path: Path) -> dict[int, str]:
    """Carrega o mapeamento dos três níveis de urgência simulada."""
    labels = pd.read_csv(path)
    mapping = dict(zip(labels["urgency_label"], labels["urgency"], strict=True))
    if len(labels) != len(LABEL_NAMES) or mapping != LABEL_NAMES:
        raise ValueError("Label catalogue does not match synthetic urgency")
    return mapping


def text_key(text: str) -> str:
    """Identifica textos equivalentes sem expor o conteúdo nos relatórios."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def text_groups(dataset: pd.DataFrame) -> pd.Series:
    """Agrupa laudos por Unicode, caixa e espaços normalizados."""
    return dataset["report_text"].map(text_key)


def scenario_groups(dataset: pd.DataFrame) -> pd.Series:
    """Agrupa a família de variações, não somente duplicatas textuais."""
    return dataset["scenario_id"]
