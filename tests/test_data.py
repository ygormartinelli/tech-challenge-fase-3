"""Testes do contrato de dados."""

import pandas as pd
import pytest

from techchallenge_fase3.data import validate_dataset


def test_validate_dataset_accepts_expected_columns() -> None:
    """Aceita dados com texto e rótulo válidos."""
    dataset = pd.DataFrame({"condition_label": [1], "medical_abstract": ["text"]})
    validate_dataset(dataset)


def test_validate_dataset_rejects_missing_columns() -> None:
    """Rejeita dados sem o campo de texto obrigatório."""
    dataset = pd.DataFrame({"condition_label": [1]})
    with pytest.raises(ValueError, match="medical_abstract"):
        validate_dataset(dataset)
