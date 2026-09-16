"""Fixtures sintéticas; a CI não depende do corpus local."""

import pandas as pd
import pytest

from techchallenge_fase3.pipelines.generate import build_datasets


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fornece os 45 cenários de treino, sem repetir suas variações."""
    train, _ = build_datasets()
    return train.drop_duplicates("scenario_id").reset_index(drop=True)
