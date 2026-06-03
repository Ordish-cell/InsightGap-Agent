from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.web_app.core.config import settings
from src.web_app.db.repositories.artifact_repository import ArtifactRepository


class ArtifactService:
    def create_artifact_record(self, user_id: int, artifact_type: str, title: str, run_id: int | None = None, db: Session | None = None, **metadata: Any) -> dict[str, Any]:
        if db:
            item = ArtifactRepository(db).create(user_id=user_id, run_id=run_id, artifact_type=artifact_type, title=title, metadata_json=metadata)
            return self._to_dict(item)
        return {"id": None, "user_id": user_id, "run_id": run_id, "artifact_type": artifact_type, "title": title}

    def list_artifacts(self, user_id: int, db: Session | None = None) -> list[dict[str, Any]]:
        if db:
            return [self._to_dict(item) for item in ArtifactRepository(db).list_by_user(user_id)]
        return []

    def get_artifact(self, artifact_id: int, user_id: int, db: Session | None = None) -> dict[str, Any]:
        if db:
            item = ArtifactRepository(db).get_by_user(user_id, artifact_id)
            if not item:
                raise ValueError("Artifact not found")
            data = self._to_dict(item)
            path = Path(item.file_path)
            data["content"] = path.read_text(encoding="utf-8") if path.exists() else ""
            return data
        return {"id": artifact_id, "user_id": user_id, "status": "mock"}

    def save_text_artifact(self, user_id: int, filename: str, content: str) -> str:
        safe_name = Path(filename).name
        path = Path(settings.artifact_storage_path) / str(user_id)
        path.mkdir(parents=True, exist_ok=True)
        target = path / safe_name
        target.write_text(content, encoding="utf-8")
        return str(target)

    def _to_dict(self, item) -> dict[str, Any]:
        return {"id": item.id, "user_id": item.user_id, "run_id": item.run_id, "artifact_type": item.artifact_type, "title": item.title, "file_path": item.file_path, "public_url": item.public_url, "metadata": item.metadata_json or {}}


artifact_service = ArtifactService()
