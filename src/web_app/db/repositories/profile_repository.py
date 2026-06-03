from sqlalchemy import select

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import UserProfile


class ProfileRepository(BaseRepository[UserProfile]):
    model = UserProfile

    def get_by_user(self, user_id: int) -> UserProfile | None:
        return self.db.execute(select(UserProfile).where(UserProfile.user_id == user_id)).scalar_one_or_none()

    def get_or_create_default(self, user_id: int) -> UserProfile:
        profile = self.get_by_user(user_id)
        if profile:
            return profile
        return self.create(user_id=user_id)
