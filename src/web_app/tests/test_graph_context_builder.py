from src.web_app.context.builder import ContextBuilder
from src.web_app.agent.runtime.planner import plan_route
from src.web_app.services.graph_context_service import GraphContextService


class FakeGraphRepository:
    def __init__(self, fail=False):
        self.fail = fail
        self.memory_calls = []
        self.project_calls = []

    def get_user_memory_context(self, **kwargs):
        self.memory_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("neo4j down")
        return [
            {
                "memory_id": "1",
                "category": "preference",
                "preview": "用户偏好 best-effort 可降级设计",
                "target_key": "qdrant",
                "importance": 0.8,
            }
        ]

    def get_project_context(self, **kwargs):
        self.project_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("neo4j down")
        return [
            {
                "labels": ["ProjectService"],
                "key": "DocumentService",
                "path": "src/web_app/services/document_service.py",
            }
        ]


def test_graph_context_service_formats_user_scoped_context(monkeypatch):
    from src.web_app.core.config import settings

    monkeypatch.setattr(settings, "enable_neo4j", True)
    monkeypatch.setattr(settings, "neo4j_context_enabled", True)
    service = GraphContextService(repository=FakeGraphRepository())

    text = service.get_context(user_id=5, query="Qdrant graph context")

    assert "User Memory Graph" in text
    assert "Project Knowledge Graph" in text
    assert "DocumentService" in text
    assert service.repository.memory_calls[0]["user_id"] == 5


def test_graph_context_failure_falls_back(monkeypatch):
    from src.web_app.core.config import settings

    monkeypatch.setattr(settings, "enable_neo4j", True)
    monkeypatch.setattr(settings, "neo4j_context_enabled", True)
    service = GraphContextService(repository=FakeGraphRepository(fail=True))

    assert service.get_context(user_id=5, query="anything") == ""
    assert service.last_debug["fallback"] is True


def test_context_builder_includes_graph_context_section():
    context = ContextBuilder().build({"task": "answer", "graph_context": "## Project Knowledge Graph\n- ProjectService: DocumentService"})

    assert "[Graph Context]" in context
    assert "DocumentService" in context


def test_project_diagnostic_query_routes_to_chat_even_if_llm_says_research():
    query = "\u4e0a\u4f20\u6587\u6863\u5931\u8d25\u5e94\u8be5\u770b\u54ea\u4e9b\u6a21\u5757\uff1f"
    plan = plan_route(
        query,
        home_intent={
            "intent": "research",
            "risk_level": "L1",
            "suggested_route_hints": ["research_agent"],
        },
    )

    assert plan["intent"] == "chat"
    assert "research_agent" not in plan["route"]
    assert plan["answer_mode"] == "project_advice"


def test_graph_context_project_diagnostic_skips_memory_and_returns_modules(monkeypatch):
    from src.web_app.core.config import settings

    monkeypatch.setattr(settings, "enable_neo4j", True)
    monkeypatch.setattr(settings, "neo4j_context_enabled", True)
    fake = FakeGraphRepository()
    service = GraphContextService(repository=fake)
    query = "\u4e0a\u4f20\u6587\u6863\u5931\u8d25\u5e94\u8be5\u770b\u54ea\u4e9b\u6a21\u5757\uff1f"

    text = service.get_context(user_id=5, query=query)

    assert "Project Diagnostic Map" in text
    assert "DocumentService" in text
    assert fake.memory_calls == []
    assert fake.project_calls[0]["terms"]
    assert service.last_debug["project_diagnostic"] is True
