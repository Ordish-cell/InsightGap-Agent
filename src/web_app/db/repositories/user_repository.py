from sqlalchemy import select
from sqlalchemy.orm import Session

from src.web_app.db.repositories.base_repository import BaseRepository
from src.web_app.models.orm import User


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        return self.db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    def get_by_user_id(self, user_id: int) -> User | None:
        return self.db.get(User, user_id)
