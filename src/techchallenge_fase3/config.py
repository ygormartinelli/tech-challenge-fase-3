"""Configurações centralizadas da aplicação."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Lê configurações locais por variáveis de ambiente."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data/raw")
    model_dir: Path = Path("models")
    model_variant: str = "optimized"

    @property
    def train_path(self) -> Path:
        """Retorna o caminho do conjunto de treino."""
        return self.data_dir / "medical_tc_train.csv"

    @property
    def test_path(self) -> Path:
        """Retorna o caminho do conjunto de teste."""
        return self.data_dir / "medical_tc_test.csv"

    @property
    def labels_path(self) -> Path:
        """Retorna o caminho do catálogo de rótulos."""
        return self.data_dir / "medical_tc_labels.csv"
