from cryptography.fernet import Fernet
import pytest
import sys
from types import ModuleType, SimpleNamespace

from src.web_app.agent.llm.context import ModelExecutionContext
from src.web_app.agent.llm.factory import build_chat_model, normalize_model_endpoint
from src.open_deep_research.utils import get_model_runtime_config
from src.web_app.core.config import get_settings
from src.web_app.models.orm import LLMConnection, LLMModel, User, UserProfile
from src.web_app.services.agent_service import prepare_agent_run
from src.web_app.services.llm_registry_service import (
    ModelSetupError, _discover, create_connection, delete_connection, list_connections, resolve_model_context, resolve_run_model_context,
    test_connection as verify_connection, update_connection, update_preferences,
)
from src.web_app.tests.db_test_utils import make_test_session


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv("MODEL_CREDENTIALS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _user(db, email: str):
    user = User(email=email, hashed_password="x")
    db.add(user)
    db.flush()
    db.add(UserProfile(user_id=user.id))
    db.commit()
    db.refresh(user)
    return user


def _active_connection(db, user_id: int, monkeypatch):
    created = create_connection(db, user_id, {
        "provider": "openai", "display_name": "Work OpenAI",
        "fields": {"api_key": "sk-secret-value"}, "model_id": "gpt-5.4-mini",
    })
    monkeypatch.setattr("src.web_app.services.llm_registry_service._discover", lambda *_: [{"model_id": "gpt-5.4-mini", "display_name": "GPT-5.4 mini"}])
    monkeypatch.setattr("src.web_app.services.llm_registry_service.build_chat_model", lambda *a, **k: SimpleNamespace(invoke=lambda *_: SimpleNamespace(content="OK")))
    verify_connection(db, user_id, {"connection_id": created["id"]})
    return list_connections(db, user_id)[0]


def test_credentials_are_encrypted_and_masked():
    db = make_test_session()
    user = _user(db, "masked@example.com")
    result = create_connection(db, user.id, {"provider": "openai", "fields": {"api_key": "sk-12345678"}, "model_id": "gpt-5.4-mini"})
    assert result["fields"].get("api_key") is None
    assert result["secrets"]["api_key"] == {"configured": True, "masked": "••••5678"}
    assert "sk-12345678" not in str(result)


def test_blank_secret_update_preserves_existing_key():
    db = make_test_session()
    user = _user(db, "preserve@example.com")
    created = create_connection(db, user.id, {"provider": "anthropic", "fields": {"api_key": "sk-ant-original"}, "model_id": "claude-sonnet-4-6"})
    updated = update_connection(db, user.id, created["id"], {"fields": {"api_key": "", "base_url": "https://example.test"}})
    assert updated["secrets"]["api_key"]["masked"].endswith("inal")
    assert updated["fields"]["base_url"] == "https://example.test"


def test_model_context_is_user_scoped_and_run_snapshot(monkeypatch):
    db = make_test_session()
    owner = _user(db, "owner@example.com")
    stranger = _user(db, "stranger@example.com")
    connection = _active_connection(db, owner.id, monkeypatch)
    model_id = connection["models"][0]["id"]
    update_preferences(db, owner.id, model_id)

    context = resolve_model_context(db, owner.id, None)
    assert context.model == "gpt-5.4-mini"
    assert context.secrets["api_key"] == "sk-secret-value"
    assert "secrets" not in context.public_dict()
    with pytest.raises(ModelSetupError, match="model_not_found"):
        resolve_model_context(db, stranger.id, model_id)

    prepared = prepare_agent_run(db, owner.id, {"user_input": "hello", "model_config_id": model_id})
    assert prepared["run_id"] > 0
    from src.web_app.models.orm import AgentRun
    run = db.get(AgentRun, prepared["run_id"])
    assert run.graph_state["model_config_id"] == model_id
    assert "secrets" not in run.graph_state["model_context"]


def test_setup_required_without_default():
    db = make_test_session()
    user = _user(db, "setup@example.com")
    with pytest.raises(ModelSetupError, match="model_setup_required"):
        resolve_model_context(db, user.id, None)


def test_existing_run_keeps_snapshot_after_connection_edit_and_soft_delete(monkeypatch):
    db = make_test_session()
    user = _user(db, "resume@example.com")
    connection = _active_connection(db, user.id, monkeypatch)
    model_id = connection["models"][0]["id"]
    snapshot = resolve_model_context(db, user.id, model_id).public_dict()

    row = db.get(LLMConnection, connection["id"])
    model = db.get(LLMModel, model_id)
    row.protocol = "openai_chat_completions"
    model.model_id = "changed-after-run-started"
    db.commit()
    delete_connection(db, user.id, connection["id"])

    restored = resolve_run_model_context(db, user.id, snapshot)
    assert restored.protocol == "openai_responses"
    assert restored.model == "gpt-5.4-mini"
    assert restored.secrets["api_key"] == "sk-secret-value"


@pytest.mark.parametrize(
    ("provider", "protocol", "config", "expected_class"),
    [
        ("openai", "openai_responses", {"base_url": "https://api.openai.test/v1"}, "ChatOpenAI"),
        ("azure_openai", "openai_chat_completions", {"endpoint": "https://azure.test", "deployment": "deployment", "api_version": "2025-04-01-preview"}, "AzureChatOpenAI"),
        ("anthropic", "anthropic_messages", {"base_url": "https://anthropic.test"}, "ChatAnthropic"),
        ("gemini", "google_generate_content", {}, "ChatGoogleGenerativeAI"),
        ("deepseek", "openai_chat_completions", {"base_url": "https://deepseek.test/v1"}, "ChatOpenAI"),
        ("qwen", "openai_responses", {"base_url": "https://qwen.test/v1"}, "ChatOpenAI"),
        ("openrouter", "openai_chat_completions", {"base_url": "https://openrouter.test/v1", "site_url": "https://app.test", "app_name": "Agent OS"}, "ChatOpenAI"),
        ("ollama", "ollama_chat", {"base_url": "http://127.0.0.1:11434"}, "ChatOpenAI"),
        ("custom", "openai_chat_completions", {"base_url": "https://custom.test/v1/chat/completions", "auth_header": "X-Token"}, "ChatOpenAI"),
    ],
)
def test_factory_normalizes_all_supported_providers(monkeypatch, provider, protocol, config, expected_class):
    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_class(name):
        return type(name, (FakeModel,), {})

    openai_module = ModuleType("langchain_openai")
    openai_module.ChatOpenAI = fake_class("ChatOpenAI")
    openai_module.AzureChatOpenAI = fake_class("AzureChatOpenAI")
    anthropic_module = ModuleType("langchain_anthropic")
    anthropic_module.ChatAnthropic = fake_class("ChatAnthropic")
    gemini_module = ModuleType("langchain_google_genai")
    gemini_module.ChatGoogleGenerativeAI = fake_class("ChatGoogleGenerativeAI")
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_module)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", anthropic_module)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", gemini_module)

    context = ModelExecutionContext(1, 1, 1, provider, protocol, "model-id", "Model", config, {"api_key": "secret"}, {"tools": True})
    result = build_chat_model(context, streaming=True)
    assert type(result).__name__ == expected_class
    assert result.kwargs["streaming"] is True
    if provider == "openai":
        assert result.kwargs["use_responses_api"] is True
    if provider == "ollama":
        assert result.kwargs["base_url"] == "http://127.0.0.1:11434/v1"
    if provider == "custom":
        assert result.kwargs["default_headers"]["X-Token"] == "secret"


def test_custom_can_save_before_model_selection_and_azure_uses_deployment():
    db = make_test_session()
    user = _user(db, "required@example.com")
    custom = create_connection(db, user.id, {"provider": "custom", "fields": {"base_url": "https://custom.test", "api_key": "secret"}})
    assert custom["status"] == "draft" and custom["models"] == []
    with pytest.raises(ModelSetupError, match="model_id_required"):
        verify_connection(db, user.id, {"connection_id": custom["id"]})
    azure = create_connection(db, user.id, {
        "provider": "azure_openai",
        "fields": {"api_key": "secret", "endpoint": "https://azure.test", "deployment": "production-chat", "api_version": "2025-04-01-preview"},
    })
    assert azure["models"][0]["model_id"] == "production-chat"


def test_custom_headers_are_encrypted_and_unknown_fields_are_rejected():
    db = make_test_session()
    user = _user(db, "headers@example.com")
    created = create_connection(db, user.id, {
        "provider": "custom",
        "model_id": "private-model",
        "fields": {"base_url": "https://custom.test", "custom_headers": {"X-Private-Token": "header-secret"}},
    })
    assert "custom_headers" not in created["fields"]
    assert created["secrets"]["custom_headers"] == {"configured": True, "masked": "••••"}
    assert "header-secret" not in str(created)
    with pytest.raises(ModelSetupError, match="unsupported_fields"):
        create_connection(db, user.id, {"provider": "ollama", "fields": {"base_url": "http://localhost:11434", "password": "leak"}})


def test_model_discovery_follows_provider_pagination(monkeypatch):
    class Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    pages = [
        {"data": [{"id": "model-a"}], "has_more": True, "last_id": "model-a"},
        {"data": [{"id": "model-b"}], "has_more": False},
    ]
    seen_params = []

    def fake_get(*_args, **kwargs):
        seen_params.append(kwargs.get("params"))
        return Response(pages.pop(0))

    monkeypatch.setattr("src.web_app.services.llm_registry_service.httpx.get", fake_get)
    result = _discover("openai_models", {"base_url": "https://provider.test/v1", "api_key": "secret"})
    assert [item["model_id"] for item in result] == ["model-a", "model-b"]
    assert seen_params == [None, {"after": "model-a"}]


def test_deep_research_stages_receive_run_scoped_connection_options():
    config = {
        "configurable": {
            "model_api_key": "run-secret",
            "model_extra": {"base_url": "https://provider.test/v1", "use_responses_api": True},
        }
    }
    result = get_model_runtime_config("openai:model-id", 2048, config)
    assert result == {
        "model": "openai:model-id",
        "max_tokens": 2048,
        "api_key": "run-secret",
        "tags": ["langsmith:nostream"],
        "base_url": "https://provider.test/v1",
        "use_responses_api": True,
    }


def test_rename_and_identical_save_keep_verification(monkeypatch):
    db = make_test_session()
    user = _user(db, "rename@example.test")
    connection = _active_connection(db, user.id, monkeypatch)
    changed = update_connection(db, user.id, connection["id"], {"display_name": "Personal", "fields": connection["fields"], "protocol": connection["protocol"]})
    assert changed["revision"] == connection["revision"]
    assert changed["status"] == "active" and changed["last_test_status"] == "passed"
    changed = update_connection(db, user.id, connection["id"], {"fields": {"base_url": "https://other.test/v1"}})
    assert changed["revision"] == connection["revision"] + 1
    assert changed["status"] == "draft"


def test_generation_test_uses_selected_protocol_and_never_requires_catalog(monkeypatch):
    db = make_test_session()
    user = _user(db, "generate@example.test")
    connection = create_connection(db, user.id, {"provider": "openai", "protocol": "openai_responses", "fields": {"api_key": "secret"}})
    captured = []
    def factory(context, **kwargs):
        captured.append(context)
        return SimpleNamespace(invoke=lambda *_: SimpleNamespace(content="OK"))
    monkeypatch.setattr("src.web_app.services.llm_registry_service.build_chat_model", factory)
    monkeypatch.setattr("src.web_app.services.llm_registry_service._discover", lambda *_: pytest.fail("Generation test must not require /models"))
    result = verify_connection(db, user.id, {"connection_id": connection["id"], "model_id": "my-text-model"})
    assert result["persisted"] and result["latency_ms"] >= 0
    assert captured[0].protocol == "openai_responses" and captured[0].model == "my-text-model"
    assert list_connections(db, user.id)[0]["models"][0]["model_id"] == "my-text-model"


def test_preview_never_verifies_unsaved_credentials(monkeypatch):
    db = make_test_session()
    user = _user(db, "preview@example.test")
    connection = create_connection(db, user.id, {"provider": "openai", "fields": {"api_key": "saved"}, "model_id": "test"})
    monkeypatch.setattr("src.web_app.services.llm_registry_service.build_chat_model", lambda *a, **k: SimpleNamespace(invoke=lambda *_: SimpleNamespace(content="OK")))
    result = verify_connection(db, user.id, {"connection_id": connection["id"], "fields": {"api_key": "different"}})
    assert result["persisted"] is False
    assert list_connections(db, user.id)[0]["status"] == "draft"


def test_stale_generation_test_cannot_activate_changed_connection(monkeypatch):
    db = make_test_session()
    user = _user(db, "stale@example.test")
    connection = create_connection(db, user.id, {"provider": "openai", "fields": {"api_key": "saved"}, "model_id": "test"})
    def invoke(*_):
        update_connection(db, user.id, connection["id"], {"fields": {"api_key": "changed"}})
        return SimpleNamespace(content="OK")
    monkeypatch.setattr("src.web_app.services.llm_registry_service.build_chat_model", lambda *a, **k: SimpleNamespace(invoke=invoke))
    with pytest.raises(ModelSetupError, match="connection_changed_during_test"):
        verify_connection(db, user.id, {"connection_id": connection["id"]})
    assert list_connections(db, user.id)[0]["status"] == "draft"


def test_discovery_works_before_verification_and_preserves_disabled_models(monkeypatch):
    from src.web_app.services.llm_registry_service import discover_models, delete_model
    db = make_test_session()
    user = _user(db, "discover@example.test")
    connection = create_connection(db, user.id, {"provider": "custom", "fields": {"base_url": "https://custom.test/v1"}, "model_id": "old"})
    delete_model(db, user.id, connection["id"], connection["models"][0]["id"])
    monkeypatch.setattr("src.web_app.services.llm_registry_service._discover", lambda *_: [{"model_id": "old"}, {"model_id": "new"}])
    result = discover_models(db, user.id, connection["id"])
    assert [item["model_id"] for item in result] == ["new"]
    assert list_connections(db, user.id)[0]["status"] == "draft"


def test_encrypted_custom_headers_reach_model_factory(monkeypatch):
    module = ModuleType("langchain_openai")
    module.ChatOpenAI = lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    context = ModelExecutionContext(1, 1, 1, "custom", "openai_chat_completions", "m", "m", {"base_url": "https://gateway.test/v1"}, {"custom_headers": {"X-Private": "encrypted-value"}})
    assert build_chat_model(context).kwargs["default_headers"]["X-Private"] == "encrypted-value"


@pytest.mark.parametrize("url", ["ftp://example.test", "http://user:secret@example.test", "https://example.test?key=secret", "not-a-url"])
def test_invalid_urls_rejected_before_network(url):
    db = make_test_session()
    user = _user(db, "url@example.test")
    with pytest.raises(ModelSetupError, match="invalid_url"):
        create_connection(db, user.id, {"provider": "custom", "fields": {"base_url": url}})


def test_discovery_respects_custom_auth_and_full_endpoint(monkeypatch):
    def get(url, **kwargs):
        assert url == "https://gateway.test/v1/models"
        assert kwargs["headers"] == {"X-Key": "secret", "X-Project": "personal"}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": [{"id": "m"}]})
    monkeypatch.setattr("src.web_app.services.llm_registry_service.httpx.get", get)
    assert _discover("openai_models", {"base_url": "https://gateway.test/v1/responses", "api_key": "secret", "auth_header": "X-Key", "custom_headers": {"X-Project": "personal"}})[0]["model_id"] == "m"


def test_first_default_is_model_actually_tested(monkeypatch):
    from src.web_app.services.llm_registry_service import add_model, get_preferences
    db = make_test_session()
    user = _user(db, "default-tested@example.test")
    connection = create_connection(db, user.id, {"provider": "openai", "fields": {"api_key": "secret"}, "model_id": "not-tested"})
    tested = add_model(db, user.id, connection["id"], {"model_id": "tested"})
    monkeypatch.setattr("src.web_app.services.llm_registry_service.build_chat_model", lambda *a, **k: SimpleNamespace(invoke=lambda *_: SimpleNamespace(content="OK")))
    verify_connection(db, user.id, {"connection_id": connection["id"], "model_id": "tested"})
    assert get_preferences(db, user.id)["default_model_config_id"] == tested["id"]


@pytest.mark.asyncio
async def test_research_model_transport_receives_encrypted_headers(monkeypatch):
    from src.web_app.agent.llm.context import use_model_context
    from src.web_app.research.open_deep_research_adapter import OpenDeepResearchAdapter
    seen = {}
    async def invoke(state, config):
        seen.update(config["configurable"])
        return {}
    module = ModuleType("open_deep_research.deep_researcher")
    module.deep_researcher = SimpleNamespace(ainvoke=invoke)
    monkeypatch.setitem(sys.modules, "open_deep_research.deep_researcher", module)
    adapter = OpenDeepResearchAdapter()
    monkeypatch.setattr(adapter, "_parse_output", lambda *args: "parsed")
    context = ModelExecutionContext(1, 1, 1, "custom", "ollama_chat", "model", "Model", {"base_url": "http://localhost:11434/v1"}, {"custom_headers": {"X-Private": "secret"}})
    with use_model_context(context):
        assert await adapter._invoke_graph("question", 1, "test", {}, [], "shallow") == "parsed"
    assert seen["model_extra"]["base_url"] == "http://localhost:11434/v1"
    assert seen["model_extra"]["default_headers"] == {"X-Private": "secret"}


@pytest.mark.parametrize("protocol,endpoint,expected", [
    ("openai_chat_completions", "https://host.test/v1/chat/completions", "https://host.test/v1"),
    ("openai_responses", "https://host.test/v1/responses", "https://host.test/v1"),
    ("anthropic_messages", "https://host.test/v1/messages", "https://host.test"),
    ("google_generate_content", "https://host.test/v1beta/models/my-model:generateContent", "https://host.test/v1beta"),
    ("ollama_chat", "http://localhost:11434/api/chat", "http://localhost:11434"),
])
def test_custom_full_endpoints_are_normalized(protocol, endpoint, expected):
    assert normalize_model_endpoint(endpoint, protocol) == expected
