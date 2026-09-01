"""Comando para exportar o modelo para ONNX Runtime."""

from techchallenge_fase3.artifacts import export_onnx, load_original
from techchallenge_fase3.config import Settings


def main() -> None:
    """Exporta o artefato original para o formato ONNX."""
    settings = Settings()
    model = load_original(settings.model_dir)
    export_onnx(model, settings.model_dir)


if __name__ == "__main__":
    main()
