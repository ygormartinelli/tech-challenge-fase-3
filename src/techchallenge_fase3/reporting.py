"""Relatórios agregados, proveniência e escrita atômica de metadados."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    """Calcula SHA-256 para identificar precisamente um arquivo."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment() -> dict[str, Any]:
    """Registra versões e plataforma sem caminhos pessoais ou segredos."""
    packages = ("scikit-learn", "numpy", "pandas", "onnxruntime", "skl2onnx")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
    }


def write_json(path: Path, value: Any) -> None:
    """Publica JSON completo; uma interrupção preserva a versão anterior."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
