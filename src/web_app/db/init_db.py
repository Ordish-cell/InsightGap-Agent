from src.web_app.db.base import Base
from src.web_app.db.session import engine
from src.web_app.models import orm


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
