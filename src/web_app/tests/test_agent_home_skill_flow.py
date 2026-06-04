import pytest

from src.web_app.models.orm import FeedCard, InfoItem, Skill, User
from src.web_app.services.agent_service import list_steps, run_agent
from src.web_app.tests.db_test_utils import make_test_session


def _user(db, email="home-agent@example.com"):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _feed_card(db, user_id: int) -> FeedCard:
    item = InfoItem(title="Agent workflow signal", summary="Research automation signal", content="agent research workflow", source_url="https://example.com/signal")
    db.add(item)
    db.commit()
    db.refresh(item)
    card = FeedCard(
        user_id=user_id,
        info_item_id=item.id,
        title="Agent workflow opportunity",
        one_sentence_value="Build reusable research workflows from feed signals.",
        why_you="Matches your Agent OS direction.",
        information_gap="Most teams do not convert signals into reusable workflows.",
        evidence=[{"title": "Signal", "url": item.source_url}],
        suggested_actions=["deep_research", "create_skill_draft"],
        score_detail={"source_type": "manual", "domain": "agent"},
        final_score=0.91,
        exposure_bucket="explicit_related",
    )
    db.add(card)
    db.commit()
    db.refresh(card)
    return card


def _skill(db, user_id: int, status="approved", trigger_text="research report workflow") -> Skill:
    skill = Skill(
        user_id=user_id,
        name="Research Report Workflow",
        description="Create research reports from reusable agent workflows.",
        trigger_text=trigger_text,
        input_schema={"query": "string"},
        context_recipe=["feed card", "memory", "research evidence"],
        tool_plan=["research", "artifact"],
        output_schema={"report": "markdown"},
        safety_level="read_only",
        eval_checks=[],
        status=status,
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)
    return skill


def test_home_chat_request_creates_agent_run_with_context():
    db = make_test_session()
    user = _user(db)
    card = _feed_card(db, user.id)

    result = run_agent(
        db,
        user.id,
        {
            "input": "research this signal",
            "source": "home_chat",
            "page_context": {"page": "home", "selected_feed_card_id": card.id},
            "create_skill_draft": False,
            "create_skill_draft_if_reusable": False,
        },
    )

    assert result["status"] == "completed"
    assert result["route"] == "research"
    steps = list_steps(db, user.id, result["run_id"])
    context_step = next(step for step in steps if step["node_name"] == "context_builder")
    assert context_step["output"]["feed_card_loaded"] is True


def test_legacy_agent_request_without_page_context_still_works():
    db = make_test_session()
    user = _user(db, "legacy-agent@example.com")

    result = run_agent(db, user.id, {"user_input": "hello there", "create_skill_draft_if_reusable": False})

    assert result["status"] == "completed"
    assert result["route"] == "memory"


def test_skill_matching_uses_approved_high_score_skill():
    db = make_test_session()
    user = _user(db, "skill-match@example.com")
    skill = _skill(db, user.id, trigger_text="research report workflow")

    result = run_agent(db, user.id, {"user_input": "please research report workflow", "route": "memory", "create_skill_draft_if_reusable": False})

    assert result["matched_skill"]["id"] == skill.id
    assert result["matched_skill"]["match_score"] >= 0.75
    steps = list_steps(db, user.id, result["run_id"])
    assert any(step["node_name"] == "skill_matcher" and step["output"]["matched_skill"] for step in steps)


def test_low_score_skill_is_not_auto_used():
    db = make_test_session()
    user = _user(db, "skill-low@example.com")
    _skill(db, user.id, trigger_text="email outreach sequence")

    result = run_agent(db, user.id, {"user_input": "summarize a note", "route": "memory", "create_skill_draft_if_reusable": False})

    assert result["matched_skill"] is None
    assert result["candidate_skills"] == []


def test_reusable_task_creates_skill_draft():
    db = make_test_session()
    user = _user(db, "draft-agent@example.com")

    result = run_agent(db, user.id, {"user_input": "以后复用这个流程 create report", "route": "artifact"})

    assert result["created_skill_draft"]
    assert result["reusable_score"] >= 0.70
    steps = list_steps(db, user.id, result["run_id"])
    assert any(step["node_name"] == "skill_draft_detector" and step["output"]["created_skill_draft"] for step in steps)


def test_casual_chat_does_not_create_skill_draft():
    db = make_test_session()
    user = _user(db, "casual-agent@example.com")

    result = run_agent(db, user.id, {"user_input": "hello", "route": "memory"})

    assert result["created_skill_draft"] is None
    assert result["reusable_score"] < 0.50


def test_feed_card_context_is_user_isolated():
    db = make_test_session()
    owner = _user(db, "feed-owner@example.com")
    other = _user(db, "feed-other@example.com")
    card = _feed_card(db, owner.id)

    result = run_agent(
        db,
        other.id,
        {
            "user_input": "research this",
            "source": "home_chat",
            "page_context": {"page": "home", "selected_feed_card_id": card.id},
            "create_skill_draft_if_reusable": False,
        },
    )

    steps = list_steps(db, other.id, result["run_id"])
    context_step = next(step for step in steps if step["node_name"] == "context_builder")
    assert context_step["output"]["feed_card_loaded"] is False


@pytest.mark.parametrize("status", ["draft", "disabled"])
def test_draft_or_disabled_skill_is_not_auto_used(status):
    db = make_test_session()
    user = _user(db, f"{status}-skill@example.com")
    _skill(db, user.id, status=status, trigger_text="research report workflow")

    result = run_agent(db, user.id, {"user_input": "research report workflow", "route": "memory", "create_skill_draft_if_reusable": False})

    assert result["matched_skill"] is None
