"""Teste de integração Docker com fixture sintética, nunca evidência de ML."""

import json
import subprocess
import tempfile
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from techchallenge_fase3.artifacts import export_onnx, save_original
from techchallenge_fase3.data import TASK_ID
from techchallenge_fase3.modeling import train_model
from techchallenge_fase3.pipelines.generate import build_datasets
from techchallenge_fase3.reporting import file_hash, write_json


def fixture_release(root: Path) -> None:
    """Publica somente uma fixture original, sem alegar aprovação de benchmark."""
    train, _ = build_datasets()
    model = train_model(train.drop_duplicates("scenario_id"))
    release = root / "releases/fixture"
    save_original(model, release)
    export_onnx(model, release)
    hashes = {path.name: file_hash(path) for path in release.iterdir()}
    write_json(
        release / "manifest.json",
        {
            "model_hashes": hashes,
            "optimized_approved": False,
            "task_id": TASK_ID,
            "data_origin": "synthetic",
        },
    )
    write_json(root / "current.json", {"release": "releases/fixture"})
    root.chmod(0o755)


def docker(*arguments: str) -> str:
    """Executa comandos com argumentos separados e falha explicitamente."""
    return subprocess.run(
        ["docker", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def wait_ready(url: str) -> None:
    """Espera readiness real dentro de uma janela limitada."""
    deadline = monotonic() + 60
    while monotonic() < deadline:
        try:
            with urlopen(url + "/health", timeout=2) as response:
                assert json.load(response)["status"] == "ok"
            return
        except (URLError, TimeoutError, ConnectionError):
            sleep(1)
    raise TimeoutError("Container did not become ready")


def verify(url: str) -> None:
    """Verifica predição, rejeição e exposição das métricas no container."""
    payload = json.dumps({"report_text": "Exame sem alterações agudas."}).encode()
    request = Request(
        url + "/predict", data=payload, headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=5) as response:
        result = json.load(response)
        assert result["urgency_label"] in range(3)
        assert result["data_origin"] == "synthetic"
        assert result["model_variant"] == "original"
    try:
        urlopen(
            Request(
                url + "/predict",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            ),
            timeout=5,
        )
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Malformed request accepted")
    with urlopen(url + "/metrics", timeout=5) as response:
        assert b"medical_classifier_requests_total" in response.read()


def main() -> None:
    """Cria e remove somente o container temporário desta verificação."""
    with tempfile.TemporaryDirectory(prefix="fase3-image-smoke-") as directory:
        root = Path(directory)
        fixture_release(root)
        container = docker(
            "run",
            "--detach",
            "--rm",
            "-p",
            "127.0.0.1::8000",
            "-v",
            f"{root.as_posix()}:/app/models:ro",
            "medical-classifier:ci",
        )
        try:
            port = docker("port", container, "8000/tcp").rsplit(":", 1)[1]
            url = f"http://127.0.0.1:{port}"
            wait_ready(url)
            verify(url)
            print("Docker API smoke test passed (synthetic fixture, original variant).")
        finally:
            docker("stop", container)


if __name__ == "__main__":
    main()
