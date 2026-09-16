"""Publicação atômica somente após avaliação e benchmark aprovados."""

import json
import shutil
from pathlib import Path
from uuid import uuid4

from techchallenge_fase3.artifacts import (
    OPTIMIZED_MODEL,
    ORIGINAL_MODEL,
    validate_release,
)
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import TASK_ID
from techchallenge_fase3.pipelines.benchmark import benchmark_accepted
from techchallenge_fase3.reporting import file_hash, write_json


def checked_evidence(settings: Settings) -> dict:
    """Rejeita relatórios antigos ou que não correspondam ao candidato."""
    report = json.loads(
        (settings.report_dir / "latency_benchmark.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (settings.report_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    for evidence in (report, evaluation):
        if (
            evidence.get("task_id") != TASK_ID
            or evidence.get("data_origin") != "synthetic"
        ):
            raise ValueError("Evidence does not describe synthetic urgency")
    if not benchmark_accepted(report):
        raise ValueError("Benchmark did not approve optimization")
    for name in (ORIGINAL_MODEL, OPTIMIZED_MODEL):
        if report["model_hashes"][name] != file_hash(settings.candidate_dir / name):
            raise ValueError("Benchmark belongs to different model artifacts")
    if evaluation["model_sha256"] != report["model_hashes"][ORIGINAL_MODEL]:
        raise ValueError("Evaluation belongs to a different model")
    if report["test_sha256"] != file_hash(settings.test_path):
        raise ValueError("Test data changed after benchmark")
    if evaluation["test_sha256"] != report["test_sha256"]:
        raise ValueError("Evaluation used different test data")
    return report


def publish(settings: Settings) -> Path:
    """Salva release imutável e troca um único ponteiro; preserva rollback."""
    report = checked_evidence(settings)
    relative = Path("releases") / uuid4().hex
    release = settings.model_dir / relative
    release.mkdir(parents=True)
    for name in (ORIGINAL_MODEL, OPTIMIZED_MODEL):
        shutil.copy2(settings.candidate_dir / name, release / name)
    for name in ("validation.json", "evaluation.json", "latency_benchmark.json"):
        shutil.copy2(settings.report_dir / name, release / name)
    write_json(release / "manifest.json", report)
    validate_release(release)
    write_json(settings.model_dir / "current.json", {"release": relative.as_posix()})
    return release


def main() -> None:
    """Publica sem reiniciar a API; a adoção ocorre no próximo restart."""
    print(f"Published: {publish(Settings())}")


if __name__ == "__main__":
    main()
