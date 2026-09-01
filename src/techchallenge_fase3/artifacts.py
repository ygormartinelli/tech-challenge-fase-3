"""Persistência e carregamento dos artefatos de inferência."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

ORIGINAL_MODEL = "classifier.joblib"
OPTIMIZED_MODEL = "classifier.onnx"


def original_path(model_dir: Path) -> Path:
    """Retorna o caminho do pipeline scikit-learn."""
    return model_dir / ORIGINAL_MODEL


def optimized_path(model_dir: Path) -> Path:
    """Retorna o caminho do modelo ONNX."""
    return model_dir / OPTIMIZED_MODEL


def save_original(model: Any, model_dir: Path) -> Path:
    """Salva o pipeline original e retorna seu caminho."""
    model_dir.mkdir(parents=True, exist_ok=True)
    target = original_path(model_dir)
    joblib.dump(model, target)
    return target


def load_original(model_dir: Path) -> Any:
    """Carrega o pipeline scikit-learn persistido."""
    return joblib.load(original_path(model_dir))


def export_onnx(model: Any, model_dir: Path) -> Path:
    """Converte o pipeline scikit-learn para ONNX Runtime."""
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier = model.named_steps["classifier"]
    options = {id(classifier): {"zipmap": False}}
    onnx_model = convert_sklearn(
        model,
        initial_types=[("medical_abstract", StringTensorType([None, 1]))],
        options=options,
    )
    target = optimized_path(model_dir)
    target.write_bytes(onnx_model.SerializeToString())
    return target


class OnnxPredictor:
    """Adaptador do ONNX Runtime para o contrato do classificador."""

    def __init__(self, path: Path) -> None:
        """Inicializa uma sessão ONNX apenas com CPU."""
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, texts: list[str]) -> np.ndarray:
        """Retorna os rótulos preditos para uma lista de textos."""
        inputs = {self.input_name: np.asarray(texts, dtype=object).reshape(-1, 1)}
        return np.asarray(self.session.run(None, inputs)[0]).reshape(-1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Retorna as probabilidades de classe para uma lista de textos."""
        inputs = {self.input_name: np.asarray(texts, dtype=object).reshape(-1, 1)}
        return np.asarray(self.session.run(None, inputs)[1])


def load_predictor(model_dir: Path, variant: str) -> tuple[Any, str]:
    """Carrega o preditor solicitado, com fallback seguro ao original."""
    if variant == "optimized" and optimized_path(model_dir).exists():
        return OnnxPredictor(optimized_path(model_dir)), "optimized"
    return load_original(model_dir), "original"
