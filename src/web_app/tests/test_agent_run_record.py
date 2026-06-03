from src.web_app.models.orm import User
from src.web_app.services.agent_service import list_steps, run_agent
from src.web_app.tests.db_test_utils import make_test_session


def test_agent_run_and_step_record():
    db = make_test_session()
    user = User(email="agent@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)

    result = run_agent(db, user.id, {"user_input": "帮我总结资料"})
    assert result["run_id"]
    assert result["status"] == "completed"
    assert list_steps(db, user.id, result["run_id"])
