"""Treino e avaliação do classificador de texto."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from techchallenge_fase3.data import (
    LABEL_NAMES,
    scenario_groups,
    text_groups,
    validate_dataset,
)


def build_pipeline() -> Pipeline:
    """Cria o pipeline leve de TF-IDF e regressão logística."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        max_features=30_000,
        sublinear_tf=False,
        token_pattern=r"[a-zA-Z0-9_]{2,}",
    )
    classifier = LogisticRegression(
        max_iter=1_000, class_weight="balanced", random_state=42
    )
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def train_model(dataset: pd.DataFrame) -> Pipeline:
    """Treina o pipeline nos textos e rótulos recebidos."""
    validate_dataset(dataset)
    if set(dataset.urgency_label) != set(LABEL_NAMES):
        raise ValueError("Training requires all three urgency classes")
    model = build_pipeline()
    return model.fit(dataset["report_text"], dataset["urgency_label"])


def evaluate_model(model: Any, dataset: pd.DataFrame) -> dict[str, Any]:
    """Calcula métricas de classificação para um dataset rotulado."""
    expected = dataset["urgency_label"]
    predicted = model.predict(dataset["report_text"].tolist())
    labels = list(LABEL_NAMES)
    details = classification_report(
        expected, predicted, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "rows": len(dataset),
        "macro_f1": float(details["macro avg"]["f1-score"]),
        "per_class": {str(label): details[str(label)] for label in labels},
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=labels
        ).tolist(),
    }


def predictions_match(first: np.ndarray, second: np.ndarray) -> bool:
    """Verifica se dois vetores de predição são equivalentes."""
    return bool(np.array_equal(first.astype(int), second.astype(int)))


def grouped_holdout(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Usa o primeiro fold estratificado por grupos, aproximadamente 80/20."""
    groups = scenario_groups(dataset)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, validation = next(splitter.split(dataset, dataset.urgency_label, groups))
    assert not set(groups.iloc[train]) & set(groups.iloc[validation])
    assert not set(text_groups(dataset.iloc[train])) & set(
        text_groups(dataset.iloc[validation])
    )
    return dataset.iloc[train], dataset.iloc[validation]


def validate_baseline(dataset: pd.DataFrame) -> dict[str, Any]:
    """Compara modelo fixo e classe majoritária sem tocar no teste fornecido."""
    train, validation = grouped_holdout(dataset)
    model = train_model(train)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train.report_text, train.urgency_label)
    return {
        "method": "Primeiro fold StratifiedGroupKFold(5, shuffle=True, seed=42)",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "shared_text_groups": 0,
        "shared_scenario_groups": 0,
        "train_scenarios": train.scenario_id.nunique(),
        "validation_scenarios": validation.scenario_id.nunique(),
        "model": evaluate_model(model, validation),
        "majority_baseline": evaluate_model(dummy, validation),
        "policy": (
            "Validação por cenário sintético. Variações do mesmo caso ficam juntas. "
            "Sem evidência clínica."
        ),
    }


def scenario_evaluation(model: Any, dataset: pd.DataFrame) -> dict[str, Any]:
    """Avalia uma redação fixa por cenário, sem inflar o suporte com variações."""
    cases = dataset.drop_duplicates("scenario_id").copy()
    predicted = model.predict(cases.report_text.tolist()).astype(int)
    cases["predicted_label"] = predicted
    errors = cases[cases.urgency_label.ne(cases.predicted_label)]
    return {
        "method": "Primeira variação determinística de cada cenário, sem votação.",
        "metrics": evaluate_model(model, cases),
        "underassigned": int((predicted < cases.urgency_label).sum()),
        "overassigned": int((predicted > cases.urgency_label).sum()),
        "urgent_as_normal": int(((predicted == 0) & cases.urgency_label.eq(2)).sum()),
        "errors": errors[["scenario_id", "urgency_label", "predicted_label"]].to_dict(
            "records"
        ),
        "caveat": "Erros sintéticos, não estimativas de segurança clínica.",
    }
