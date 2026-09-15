"""Regressões dos riscos efetivamente encontrados na EDA."""

import pandas as pd
import pytest

from techchallenge_fase3.analysis import describe_corpus, overlap_summary, profile
from techchallenge_fase3.data import text_groups, text_key, validate_dataset
from techchallenge_fase3.modeling import grouped_holdout, validate_baseline


@pytest.mark.parametrize(
    "label,text",
    [
        (6, "valid text"),
        (1.5, "valid text"),
        (True, "valid text"),
        ("1", "valid text"),
        (1, "  "),
        (1, None),
        (1, 123),
    ],
)
def test_invalid_values_rejected(label: object, text: object) -> None:
    """Não permite classes inválidas nem abstracts ausentes ou não textuais."""
    with pytest.raises(ValueError):
        validate_dataset(
            pd.DataFrame({"condition_label": [label], "medical_abstract": [text]})
        )


def test_groups_normalize_case_whitespace_and_unicode() -> None:
    """Mesmas palavras não atravessam partições por diferenças cosméticas."""
    assert text_key("  TUMOR\tcell ") == text_key("tumor cell")
    assert text_key("café") == text_key("cafe\u0301")


def test_conflicting_labels_are_not_silently_deduplicated() -> None:
    """Conta conflito por texto, mesmo sem linhas integralmente repetidas."""
    data = pd.DataFrame(
        {"condition_label": [1, 2], "medical_abstract": ["Tumor cell", " tumor   cell"]}
    )
    summary = profile(data)
    assert summary["exact_duplicate_rows"] == 0
    assert summary["conflicting_text_groups"] == 1
    assert summary["rows_in_conflicting_groups"] == 2
    overlap = overlap_summary(data.iloc[:1], data.iloc[1:])
    assert overlap["test_rows_seen_in_train"] == 1
    assert overlap["single_label_empirical_accuracy_ceiling"] == 0.5


def test_grouped_validation_has_no_shared_texts(sample_data: pd.DataFrame) -> None:
    """Duplicatas e conflitos ficam juntos, com resultado determinístico."""
    data = pd.concat([sample_data, sample_data.iloc[:5]], ignore_index=True)
    train, validation = grouped_holdout(data)
    assert not set(text_groups(train)) & set(text_groups(validation))
    assert len(train) + len(validation) == len(data)
    assert set(validation.condition_label) == {1, 2, 3, 4, 5}
    assert train.index.equals(grouped_holdout(data)[0].index)


def test_analysis_and_majority_baseline_execute(sample_data: pd.DataFrame) -> None:
    """Executa a EDA real em fixture, sem substituir dados da entrega."""
    report = describe_corpus(sample_data, sample_data.iloc[:5])
    assert report["overlap"]["test_seen_share"] == 1
    assert sum(row["train"] for row in report["classes"]) == len(sample_data)
    validation = validate_baseline(sample_data)
    assert validation["model"]["macro_f1"] > validation["majority_baseline"]["macro_f1"]
