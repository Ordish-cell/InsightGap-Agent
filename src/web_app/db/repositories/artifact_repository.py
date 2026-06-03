from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import Artifact


class ArtifactRepository(BaseRepository[Artifact]):
    model = Artifact

    def list_by_user(self, user_id: int) -> list[Artifact]:
        return list(self.db.execute(select(Artifact).where(Artifact.user_id == user_id).order_by(Artifact.created_at.desc())).scalars())

    def get_by_user(self, user_id: int, artifact_id: int) -> Artifact | None:
        return self.db.execute(select(Artifact).where(Artifact.user_id == user_id, Artifact.id == artifact_id)).scalar_one_or_none()
