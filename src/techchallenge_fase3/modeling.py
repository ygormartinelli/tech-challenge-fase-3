"""Treino e avaliação do classificador de texto."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """Cria o pipeline leve de TF-IDF e regressão logística."""
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), max_features=30_000, sublinear_tf=True
    )
    classifier = LogisticRegression(max_iter=1_000, class_weight="balanced")
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def train_model(dataset: pd.DataFrame) -> Pipeline:
    """Treina o pipeline nos textos e rótulos recebidos."""
    model = build_pipeline()
    return model.fit(dataset["medical_abstract"], dataset["condition_label"])


def evaluate_model(model: Any, dataset: pd.DataFrame) -> dict[str, float]:
    """Calcula métricas de classificação para um dataset rotulado."""
    expected = dataset["condition_label"]
    predicted = model.predict(dataset["medical_abstract"])
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
    }


def predictions_match(first: np.ndarray, second: np.ndarray) -> bool:
    """Verifica se dois vetores de predição são equivalentes."""
    return bool(np.array_equal(first.astype(int), second.astype(int)))
