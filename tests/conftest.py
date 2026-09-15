"""Fixtures sintéticas; a CI não depende do corpus local."""

import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fornece grupos suficientes em cada uma das cinco categorias."""
    terms = {
        1: "cancer tumor",
        2: "bowel liver",
        3: "brain nerve",
        4: "heart coronary",
        5: "general syndrome",
    }
    return pd.DataFrame(
        [
            {
                "condition_label": label,
                "medical_abstract": f"{text} study number{index}",
            }
            for label, text in terms.items()
            for index in range(15)
        ]
    )
