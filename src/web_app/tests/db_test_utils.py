from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.web_app.db.base import Base


def make_test_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


def configure_test_model(db, user):
    """Provision the explicit model prerequisite for service behavior tests.

    Registry/setup tests deliberately do not call this helper. Providers remain
    mocked by each test; the reserved endpoint cannot target a real account.
    """
    from sqlalchemy import select
    from src.web_app.models.orm import LLMConnection, LLMModel, UserProfile
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    if profile is None:
        profile = UserProfile(user_id=user.id)
        db.add(profile)
    if profile.default_llm_model_id:
        return
    connection = LLMConnection(user_id=user.id, provider="custom", protocol="openai_chat_completions",
        display_name="Isolated test provider", status="active", last_test_status="passed",
        config_json={"base_url": "http://provider.invalid/v1"})
    db.add(connection)
    db.flush()
    model = LLMModel(connection_id=connection.id, model_id="test-model", display_name="Test model",
        capabilities_json={"tools": True, "structured_output": True, "streaming": True})
    db.add(model)
    db.flush()
    profile.default_llm_model_id = model.id
    db.commit()
