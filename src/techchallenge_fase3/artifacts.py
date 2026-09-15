"""Factory e estratégias de inferência com artefatos explicitamente aprovados."""

import json
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from techchallenge_fase3.reporting import file_hash

ORIGINAL_MODEL = "classifier.joblib"
OPTIMIZED_MODEL = "classifier.onnx"


def original_path(model_dir: Path) -> Path:
    """Retorna o caminho do pipeline scikit-learn."""
    return model_dir / ORIGINAL_MODEL


def optimized_path(model_dir: Path) -> Path:
    """Retorna o caminho do modelo ONNX."""
    return model_dir / OPTIMIZED_MODEL


def save_original(model: Any, model_dir: Path) -> Path:
    """Salva um candidato sem alterar a versão servida."""
    model_dir.mkdir(parents=True, exist_ok=True)
    target = original_path(model_dir)
    joblib.dump(model, target)
    return target


def load_original(model_dir: Path) -> Any:
    """Carrega somente artefatos locais confiáveis; joblib executa código."""
    return joblib.load(original_path(model_dir))


def export_onnx(model: Any, model_dir: Path) -> Path:
    """Exporta probabilidades densas em ordem de classes explícita."""
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier = model.named_steps["classifier"]
    vectorizer = model.named_steps["vectorizer"]
    onnx_model = convert_sklearn(
        model,
        initial_types=[("medical_abstract", StringTensorType([None, 1]))],
        options={
            id(classifier): {"zipmap": False},
            id(vectorizer): {"tokenexp": vectorizer.token_pattern},
        },
        target_opset=17,
    )
    target = optimized_path(model_dir)
    target.write_bytes(onnx_model.SerializeToString())
    return target


class Predictor(Protocol):
    """Strategy: rótulos e probabilidades em uma única inferência."""

    def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Retorna rótulos e probabilidades para a mesma passagem."""


class OriginalPredictor:
    """Adapta scikit-learn sem repetir a vetorização."""

    def __init__(self, model: Any) -> None:
        """Mantém o pipeline original e sua ordem de classes."""
        self.model = model

    def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Calcula probabilidades e recupera os rótulos correspondentes."""
        probabilities = self.model.predict_proba(texts)
        return self.model.classes_[probabilities.argmax(axis=1)], probabilities


class OnnxPredictor:
    """Executa ONNX em CPU com uma thread, apropriada à requisição individual."""

    def __init__(self, path: Path) -> None:
        """Evita pools excessivos de threads para este modelo pequeno."""
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Retorna ambos os resultados sem repetir a sessão."""
        inputs = {self.input_name: np.asarray(texts, dtype=object).reshape(-1, 1)}
        labels, probabilities = self.session.run(None, inputs)
        return np.asarray(labels).reshape(-1), np.asarray(probabilities)

    def predict(self, texts: list[str]) -> np.ndarray:
        """Atende ao contrato usado pela avaliação scikit-learn."""
        return self.predict_batch(texts)[0]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Expõe probabilidades para testes e análises."""
        return self.predict_batch(texts)[1]


def published_directory(model_dir: Path) -> Path:
    """Resolve apenas releases internas, impedindo escape de diretório."""
    pointer = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    release = (model_dir / pointer["release"]).resolve()
    if not release.is_relative_to((model_dir / "releases").resolve()):
        raise ValueError("Invalid release path")
    return release


def validate_release(release: Path) -> dict[str, Any]:
    """Confere integridade e vínculo entre aprovação e os dois modelos."""
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    for name in (ORIGINAL_MODEL, OPTIMIZED_MODEL):
        if file_hash(release / name) != manifest["model_hashes"][name]:
            raise ValueError("Model integrity check failed")
    return manifest


def load_predictor(model_dir: Path, variant: str) -> tuple[Predictor, str]:
    """Factory: ONNX exige aprovação; o original é o fallback explícito."""
    release = published_directory(model_dir)
    manifest = validate_release(release)
    if variant not in {"original", "optimized"}:
        raise ValueError("Unknown model variant")
    if variant == "optimized" and manifest["optimized_approved"]:
        return OnnxPredictor(optimized_path(release)), "optimized"
    return OriginalPredictor(load_original(release)), "original"
