"""Testes do contrato de dados."""

import pandas as pd
import pytest

from techchallenge_fase3.data import validate_dataset


def test_validate_dataset_accepts_expected_columns(sample_data: pd.DataFrame) -> None:
    """Aceita dados com texto e rótulo válidos."""
    validate_dataset(sample_data)


def test_validate_dataset_rejects_missing_columns() -> None:
    """Rejeita dados sem o campo de texto obrigatório."""
    dataset = pd.DataFrame({"urgency_label": [1]})
    with pytest.raises(ValueError, match="report_text"):
        validate_dataset(dataset)
