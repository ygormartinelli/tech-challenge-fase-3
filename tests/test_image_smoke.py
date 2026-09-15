"""Regressões da espera de inicialização do teste Docker."""

import importlib.util
from http.client import RemoteDisconnected
from io import BytesIO
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "image_smoke", Path(__file__).resolve().parents[1] / "scripts/smoke_image.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_startup_retries_closed_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Porta publicada não garante que o servidor já responde a HTTP."""
    attempts = iter([RemoteDisconnected(), BytesIO(b'{"status": "ok"}')])
    waits = []

    def open_next(*args: object, **kwargs: object) -> BytesIO:
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(smoke, "urlopen", open_next)
    monkeypatch.setattr(smoke, "sleep", waits.append)
    smoke.wait_ready("http://localhost")
    assert waits == [1]


def test_startup_has_bounded_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uma falha persistente nunca se transforma em espera infinita."""
    clock = iter([0, 61])
    monkeypatch.setattr(smoke, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="did not become ready"):
        smoke.wait_ready("http://localhost")
