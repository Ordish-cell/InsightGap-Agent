from src.web_app.models.orm import User
from src.web_app.services.agent_service import build_user_facing_answer, clear_conversation, get_conversation, list_steps, run_agent
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


def test_agent_run_persists_conversation_messages_and_answer():
    db = make_test_session()
    user = User(email="agent-conversation@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)

    result = run_agent(db, user.id, {"user_input": "hello", "source": "home_chat"})

    assert result["conversation_id"]
    assert result["thread_id"].endswith(result["conversation_id"])
    assert result["answer"]
    assert result["final_response"]["answer"] == result["answer"]
    assert result["assistant_message"]["content"] == result["answer"]

    conversation = get_conversation(db, user.id, result["conversation_id"])
    assert conversation["message_count"] == 2
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][1]["content"] == result["answer"]

    cleared = clear_conversation(db, user.id, result["conversation_id"])
    assert cleared["cleared_messages"] == 2
    assert list_steps(db, user.id, result["run_id"])


def test_generic_runtime_completed_is_not_user_answer():
    answer = build_user_facing_answer(
        {
            "user_input": "你好，你是谁？？",
            "status": "completed",
            "route_plan": {"intent": "chat", "risk_level": "L0"},
            "final_payload": {"answer": "Agent runtime completed."},
            "final_answer": "Agent runtime completed.",
            "final_output": "Agent runtime completed.",
        }
    )

    assert "信息差 Agent OS 助手" in answer
    assert answer != "Agent runtime completed."
