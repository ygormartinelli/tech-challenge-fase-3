"""EDA reproduzível do corpus, sem publicar abstracts individuais."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from techchallenge_fase3.data import LABEL_NAMES, text_groups


def length_summary(values: pd.Series) -> dict[str, float]:
    """Resume a cauda e a tendência central sem descartar extremos."""
    return {
        str(key): float(value)
        for key, value in values.describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
        ).items()
    }


def profile(dataset: pd.DataFrame) -> dict[str, Any]:
    """Descreve o grão linha-texto-rótulo e a qualidade observada."""
    groups = text_groups(dataset)
    counts = dataset.assign(key=groups).groupby("key").condition_label.nunique()
    return {
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "dtypes": {name: str(dtype) for name, dtype in dataset.dtypes.items()},
        "nulls": {name: int(n) for name, n in dataset.isna().sum().items()},
        "exact_duplicate_rows": int(dataset.duplicated().sum()),
        "unique_texts": int(groups.nunique()),
        "duplicate_text_rows_beyond_first": int(groups.duplicated().sum()),
        "conflicting_text_groups": int(counts.gt(1).sum()),
        "rows_in_conflicting_groups": int(
            groups.isin(counts[counts.gt(1)].index).sum()
        ),
        "characters": length_summary(dataset.medical_abstract.str.len()),
        "words": length_summary(dataset.medical_abstract.str.split().str.len()),
        "outside_api_length": int(
            (~dataset.medical_abstract.str.len().between(10, 20000)).sum()
        ),
    }


def class_summary(train: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, Any]]:
    """Compara suportes e proporções com o mesmo denominador por partição."""
    rows = []
    for label, name in LABEL_NAMES.items():
        counts = [int(frame.condition_label.eq(label).sum()) for frame in (train, test)]
        rows.append(
            {
                "label": label,
                "name": name,
                "train": counts[0],
                "test": counts[1],
                "train_share": counts[0] / len(train),
                "test_share": counts[1] / len(test),
                "share_difference_pp": 100
                * (counts[1] / len(test) - counts[0] / len(train)),
            }
        )
    return rows


def overlap_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Mede exposição do teste e ambiguidade do corpus combinado."""
    train_keys, test_keys = text_groups(train), text_groups(test)
    combined = pd.concat([train, test], ignore_index=True)
    groups = combined.assign(key=text_groups(combined)).groupby("key")
    label_counts = groups.condition_label.value_counts().unstack(fill_value=0)
    overlap = test_keys.isin(set(train_keys))
    return {
        "test_rows_seen_in_train": int(overlap.sum()),
        "test_seen_share": float(overlap.mean()),
        "test_unseen_rows": int((~overlap).sum()),
        "shared_text_groups": len(set(train_keys) & set(test_keys)),
        "combined_unique_texts": len(label_counts),
        "combined_conflicting_groups": int(label_counts.gt(0).sum(axis=1).gt(1).sum()),
        "single_label_empirical_accuracy_ceiling": float(
            label_counts.max(axis=1).sum() / len(combined)
        ),
        "label_cooccurrence": cooccurrence(label_counts),
    }


def cooccurrence(counts: pd.DataFrame) -> list[list[int]]:
    """Conta grupos de texto com cada par de rótulos, sem presumir causas."""
    presence = counts.reindex(columns=LABEL_NAMES, fill_value=0).gt(0).astype(int)
    return (presence.T @ presence).to_numpy().tolist()


def distribution_checks(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Compara forma dos comprimentos; o p-valor é apenas exploratório."""
    train_lengths = train.medical_abstract.str.len()
    test_lengths = test.medical_abstract.str.len()
    result = ks_2samp(train_lengths, test_lengths)
    return {
        "character_length_ks_statistic": float(result.statistic),
        "character_length_ks_pvalue": float(result.pvalue),
        "caveat": "Duplicatas violam independência; KS apenas exploratório.",
        "temporal_analysis": "Sem datas, pacientes, hospitais ou origem por linha.",
    }


def lexical_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Ajusta vocabulário somente no treino e mede cobertura no teste."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2)
    train_matrix = vectorizer.fit_transform(train.medical_abstract)
    test_matrix = vectorizer.transform(test.medical_abstract)
    features = vectorizer.get_feature_names_out()
    terms = {}
    for label in LABEL_NAMES:
        means = np.asarray(
            train_matrix[train.condition_label.eq(label).to_numpy()].mean(axis=0)
        ).ravel()
        terms[str(label)] = features[np.argsort(means)[-12:][::-1]].tolist()
    return {
        "vocabulary_size": len(features),
        "top_mean_tfidf_terms_by_class": terms,
        "train_sparsity": 1 - train_matrix.nnz / np.prod(train_matrix.shape),
        "test_zero_vectors": int((test_matrix.getnnz(axis=1) == 0).sum()),
        "test_oov_token_share": oov_share(vectorizer, test.medical_abstract),
        "near_duplicates": nearest_texts(train, test, train_matrix, test_matrix),
    }


def oov_share(vectorizer: TfidfVectorizer, texts: pd.Series) -> float:
    """Mede ocorrências fora do vocabulário limitado, após stopwords."""
    analyzer = vectorizer.build_analyzer()
    total = unknown = 0
    for text in texts:
        tokens = analyzer(text)
        total += len(tokens)
        unknown += sum(token not in vectorizer.vocabulary_ for token in tokens)
    return unknown / total if total else 0.0


def nearest_texts(
    train: pd.DataFrame, test: pd.DataFrame, train_matrix: Any, test_matrix: Any
) -> dict[str, Any]:
    """Investiga similaridade lexical, excluindo o overlap exato do contador."""
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbors.fit(train_matrix)
    distances, _ = neighbors.kneighbors(test_matrix)
    scores = np.clip(1 - distances.ravel(), 0, 1)
    unseen = ~text_groups(test).isin(set(text_groups(train))).to_numpy()
    return {
        "method": "Cosseno TF-IDF, 20000 unigramas, min_df=2; fit no treino.",
        "unseen_test_rows_similarity_ge_090": int(((scores >= 0.90) & unseen).sum()),
        "unseen_test_rows_similarity_ge_095": int(((scores >= 0.95) & unseen).sum()),
        "unseen_test_similarity_quantiles": np.quantile(
            scores[unseen], [0, 0.5, 0.9, 0.95, 1]
        ).tolist()
        if unseen.any()
        else [],
        "caveat": "Similaridade lexical não prova identidade de pacientes.",
    }


def describe_corpus(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Executa as verificações estruturais, distributivas e lexicais."""
    return {
        "train": profile(train),
        "test": profile(test),
        "classes": class_summary(train, test),
        "overlap": overlap_summary(train, test),
        "distribution": distribution_checks(train, test),
        "lexical": lexical_summary(train, test),
    }
