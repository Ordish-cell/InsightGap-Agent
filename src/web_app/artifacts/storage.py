from pathlib import Path

from src.web_app.core.config import settings


def artifact_path(filename: str) -> Path:
    base = Path(settings.artifact_storage_path)
    base.mkdir(parents=True, exist_ok=True)
    return base / filename
