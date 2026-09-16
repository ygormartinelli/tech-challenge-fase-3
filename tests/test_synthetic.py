"""Contrato e isolamento da adaptação acadêmica autorizada."""

from pathlib import Path

import pandas as pd
import pytest

from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import load_label_names, validate_dataset, validate_splits
from techchallenge_fase3.pipelines.generate import build_datasets, generate


def test_generation_is_deterministic_and_scenario_disjoint(tmp_path: Path) -> None:
    """As 3300 linhas são 75 cenários; não 3300 pacientes independentes."""
    settings = Settings(
        data_dir=tmp_path / "synthetic", report_dir=tmp_path / "reports"
    )
    first, second = generate(settings), generate(settings)
    assert first == second
    assert (first["train_rows"], first["test_rows"]) == (2700, 600)
    assert (first["train_scenarios"], first["test_scenarios"]) == (45, 30)
    assert first["clinically_validated"] is False
    assert load_label_names(settings.labels_path) == {
        0: "normal",
        1: "atenção",
        2: "urgente",
    }
    train, test = build_datasets()
    validate_splits(train, test)
    assert train.groupby("source_theme").urgency_label.nunique().eq(3).all()
    assert not train.report_text.str.contains(
        "normal|atenção|urgente", case=False
    ).any()


def test_scenario_leakage_is_rejected() -> None:
    """Mesmo uma redação diferente não pode compartilhar o cenário-base."""
    train, test = build_datasets()
    test.loc[0, "scenario_id"] = train.iloc[0].scenario_id
    with pytest.raises(ValueError, match="scenario leakage"):
        validate_splits(train, test)


def test_real_origin_and_scenario_conflicts_rejected(sample_data: pd.DataFrame) -> None:
    """Bloqueia mistura silenciosa de origens e rótulos da mesma família."""
    data = sample_data.copy()
    data.loc[0, "source_type"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        validate_dataset(data)
    data = pd.concat([sample_data, sample_data.iloc[:1]], ignore_index=True)
    data.loc[len(data) - 1, "urgency_label"] = 2
    with pytest.raises(ValueError, match="Conflicting labels"):
        validate_dataset(data)
