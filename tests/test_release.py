"""Integridade de artefatos, fallback e proteção da publicação anterior."""

import json
from pathlib import Path

import pandas as pd
import pytest

from techchallenge_fase3.artifacts import export_onnx, load_predictor, save_original
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import TASK_ID
from techchallenge_fase3.modeling import train_model
from techchallenge_fase3.pipelines.publish import publish
from techchallenge_fase3.reporting import file_hash, write_json


@pytest.fixture
def candidate(tmp_path: Path, sample_data: pd.DataFrame) -> Settings:
    """Usa tempos sintéticos apenas para testar o gate, não para alegar ganho."""
    settings = Settings(
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "raw",
        report_dir=tmp_path / "reports",
    )
    settings.data_dir.mkdir()
    sample_data.to_csv(settings.test_path, index=False)
    model = train_model(sample_data)
    save_original(model, settings.candidate_dir)
    export_onnx(model, settings.candidate_dir)
    hashes = {path.name: file_hash(path) for path in settings.candidate_dir.iterdir()}
    report = {
        "task_id": TASK_ID,
        "data_origin": "synthetic",
        "model_hashes": hashes,
        "optimized_approved": True,
        "test_sha256": file_hash(settings.test_path),
        "equivalence": {"passed": True},
        "latency": {
            "rounds": [{"original": {"p50_ms": 2}, "optimized": {"p50_ms": 1}}] * 3
        },
    }
    write_json(settings.report_dir / "latency_benchmark.json", report)
    write_json(
        settings.report_dir / "evaluation.json",
        {
            "task_id": TASK_ID,
            "data_origin": "synthetic",
            "model_sha256": hashes["classifier.joblib"],
            "test_sha256": report["test_sha256"],
        },
    )
    write_json(settings.report_dir / "validation.json", {"fixture": True})
    return settings


def test_published_models_load_and_tampering_is_rejected(candidate: Settings) -> None:
    """Modelos válidos carregam; corrupção causa falha fechada."""
    release = publish(candidate)
    predictor, variant = load_predictor(candidate.model_dir, "optimized")
    assert variant == "optimized"
    assert predictor.predict_batch(["Exame sem alterações."])[1].shape == (1, 3)
    manifest = json.loads((release / "manifest.json").read_text())
    manifest["optimized_approved"] = False
    write_json(release / "manifest.json", manifest)
    assert load_predictor(candidate.model_dir, "optimized")[1] == "original"
    (release / "classifier.onnx").write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="integrity"):
        load_predictor(candidate.model_dir, "optimized")


def test_failed_candidate_does_not_replace_release(candidate: Settings) -> None:
    """Relatório não reutiliza aprovação de outro modelo."""
    publish(candidate)
    before = (candidate.model_dir / "current.json").read_bytes()
    (candidate.candidate_dir / "classifier.onnx").write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="different model artifacts"):
        publish(candidate)
    assert (candidate.model_dir / "current.json").read_bytes() == before


def test_missing_model_and_path_escape_are_rejected(tmp_path: Path) -> None:
    """Ponteiro não pode sair do diretório de releases."""
    with pytest.raises(FileNotFoundError):
        load_predictor(tmp_path, "optimized")
    write_json(tmp_path / "current.json", {"release": "../outside"})
    with pytest.raises(ValueError, match="Invalid release path"):
        load_predictor(tmp_path, "optimized")


def test_legacy_disease_model_cannot_serve_urgency(candidate: Settings) -> None:
    """Uma release antiga não se transforma em urgência por renomear a API."""
    release = publish(candidate)
    manifest = json.loads((release / "manifest.json").read_text())
    manifest.pop("task_id")
    write_json(release / "manifest.json", manifest)
    with pytest.raises(ValueError, match="task"):
        load_predictor(candidate.model_dir, "optimized")
