from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from time import perf_counter
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.web_app.agent.llm.context import ModelExecutionContext
from src.web_app.agent.llm.crypto import decrypt_secrets, encrypt_secrets, masked_secret
from src.web_app.agent.llm.factory import build_chat_model, normalize_model_endpoint
from src.web_app.agent.llm.registry import DEFAULT_CAPABILITIES, SECRET_FIELD_KEYS, catalog, get_provider
from src.web_app.core.config import get_settings
from src.web_app.models.orm import AgentConversation, LLMConnection, LLMModel, UserProfile


class ModelSetupError(ValueError):
    pass


def get_catalog() -> list[dict[str, Any]]:
    return catalog()


def list_connections(db: Session, user_id: int) -> list[dict[str, Any]]:
    rows = db.scalars(select(LLMConnection).where(LLMConnection.user_id == user_id, LLMConnection.deleted_at.is_(None)).order_by(LLMConnection.updated_at.desc())).all()
    return [_connection_response(db, row) for row in rows]


def create_connection(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    provider = get_provider(str(payload.get("provider") or ""))
    protocol = str(payload.get("protocol") or provider.protocol)
    if protocol not in (provider.protocols or (provider.protocol,)):
        raise ModelSetupError("protocol_not_supported")
    fields = dict(payload.get("fields") or {})
    _apply_field_defaults(provider, fields)
    _validate_fields(provider, fields)
    secrets, config = _split_fields(fields)
    model_id = str(payload.get("model_id") or fields.get("deployment") or "").strip()
    row = LLMConnection(
        user_id=user_id,
        provider=provider.key,
        protocol=protocol,
        display_name=str(payload.get("display_name") or provider.label).strip(),
        config_json=config,
        encrypted_secrets=encrypt_secrets(secrets),
        status="draft",
    )
    db.add(row)
    db.flush()
    for model_payload in payload.get("models") or []:
        _add_model_row(db, row, model_payload)
    if model_id:
        source = "preset" if any(item.model_id == model_id for item in provider.models) else "manual"
        _add_model_row(db, row, {"model_id": model_id, "display_name": payload.get("model_display_name") or model_id, "source": source})
    db.commit()
    db.refresh(row)
    return _connection_response(db, row)


def update_connection(db: Session, user_id: int, connection_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    row = _connection(db, user_id, connection_id)
    provider = get_provider(row.provider)
    fields = {**(row.config_json or {}), **dict(payload.get("fields") or {})}
    old_secrets = decrypt_secrets(row.encrypted_secrets)
    incoming_secrets = {key: value for key, value in fields.items() if key in SECRET_FIELD_KEYS and value not in (None, "")}
    for key in SECRET_FIELD_KEYS:
        fields.pop(key, None)
    _apply_field_defaults(provider, fields)
    validated = {**fields, **old_secrets, **incoming_secrets}
    _validate_fields(provider, validated)
    fields = {key: value for key, value in validated.items() if key not in SECRET_FIELD_KEYS}
    next_secrets = {**old_secrets, **incoming_secrets}
    protocol = str(payload.get("protocol") or row.protocol)
    if protocol not in (provider.protocols or (provider.protocol,)):
        raise ModelSetupError("protocol_not_supported")
    changed = fields != (row.config_json or {}) or next_secrets != old_secrets or protocol != row.protocol
    row.config_json = fields
    if next_secrets != old_secrets:
        row.encrypted_secrets = encrypt_secrets(next_secrets)
    if payload.get("display_name") is not None:
        row.display_name = str(payload["display_name"]).strip()
    if payload.get("protocol") is not None:
        protocol = str(payload["protocol"])
        if protocol not in (provider.protocols or (provider.protocol,)):
            raise ModelSetupError("protocol_not_supported")
        row.protocol = protocol
    if changed:
        row.revision += 1
        row.status = "draft"
        row.last_test_status = "untested"
        row.last_test_error = ""
        row.last_tested_at = None
    db.commit()
    db.refresh(row)
    return _connection_response(db, row)


def delete_connection(db: Session, user_id: int, connection_id: int) -> None:
    row = _connection(db, user_id, connection_id)
    row.deleted_at = datetime.now(UTC).replace(tzinfo=None)
    row.status = "deleted"
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    model_ids = set(db.scalars(select(LLMModel.id).where(LLMModel.connection_id == row.id)).all())
    if profile and profile.default_llm_model_id in model_ids:
        profile.default_llm_model_id = None
    db.commit()


def add_model(db: Session, user_id: int, connection_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    row = _connection(db, user_id, connection_id)
    model = _add_model_row(db, row, payload)
    db.commit()
    db.refresh(model)
    return _model_response(model)


def update_model(db: Session, user_id: int, connection_id: int, model_pk: int, payload: dict[str, Any]) -> dict[str, Any]:
    _connection(db, user_id, connection_id)
    model = db.scalar(select(LLMModel).where(LLMModel.id == model_pk, LLMModel.connection_id == connection_id))
    if not model:
        raise ModelSetupError("model_not_found")
    if "model_id" in payload:
        next_model_id = str(payload.get("model_id") or "").strip()
        if not next_model_id:
            raise ModelSetupError("model_id_required")
        duplicate = db.scalar(select(LLMModel.id).where(LLMModel.connection_id == connection_id, LLMModel.model_id == next_model_id, LLMModel.id != model_pk))
        if duplicate:
            raise ModelSetupError("model_id_already_exists")
        payload["model_id"] = next_model_id
    if payload.get("source") not in (None, "preset", "discovered", "manual"):
        raise ModelSetupError("invalid_model_source")
    for key in ("model_id", "display_name", "source", "enabled"):
        if key in payload:
            setattr(model, key, payload[key])
    if "capabilities" in payload:
        model.capabilities_json = dict(payload["capabilities"] or {})
    db.commit()
    db.refresh(model)
    return _model_response(model)


def delete_model(db: Session, user_id: int, connection_id: int, model_pk: int) -> None:
    _connection(db, user_id, connection_id)
    model = db.scalar(select(LLMModel).where(LLMModel.id == model_pk, LLMModel.connection_id == connection_id))
    if not model:
        raise ModelSetupError("model_not_found")
    model.enabled = False
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if profile and profile.default_llm_model_id == model.id:
        profile.default_llm_model_id = None
    db.commit()


def test_connection(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Test the chosen generation protocol; a model listing is not a generation test."""
    connection_id = payload.get("connection_id")
    row = _connection(db, user_id, int(connection_id)) if connection_id else None
    provider_key = row.provider if row else str(payload.get("provider") or "")
    provider = get_provider(provider_key)
    protocol = str(payload.get("protocol") or (row.protocol if row else provider.protocol))
    stored_fields = {**(row.config_json or {}), **decrypt_secrets(row.encrypted_secrets)} if row else {}
    fields = {**stored_fields, **{k: v for k, v in dict(payload.get("fields") or {}).items() if v not in (None, "")}}
    _apply_field_defaults(provider, fields)
    if protocol not in (provider.protocols or (provider.protocol,)):
        raise ModelSetupError("protocol_not_supported")
    _validate_fields(provider, fields)
    selected_model = str(payload.get("model_id") or fields.get("deployment") or "").strip()
    if not selected_model and row:
        selected_model = db.scalar(select(LLMModel.model_id).where(LLMModel.connection_id == row.id, LLMModel.enabled.is_(True)).order_by(LLMModel.id)) or ""
    if not selected_model:
        raise ModelSetupError("model_id_required")
    # Testing edited, unsaved fields is a preview and must never activate saved credentials.
    persist = bool(row and fields == stored_fields and protocol == row.protocol)
    revision = row.revision if row else 0
    db.rollback()  # Release the read transaction before calling the provider.
    secrets, config = _split_fields(fields)
    temporary = ModelExecutionContext(0, int(connection_id or 0), revision, provider_key, protocol, selected_model, "Connection test", config, secrets, dict(DEFAULT_CAPABILITIES))
    started = perf_counter()
    failure = None
    try:
        reply = build_chat_model(temporary, temperature=None, timeout_seconds=20).invoke("Reply with OK. Do not use tools.")
        if not getattr(reply, "content", None):
            raise ModelSetupError("empty_model_response")
    except Exception as exc:
        failure = _classify_error(exc)
    if persist:
        row = db.scalar(select(LLMConnection).where(LLMConnection.id == connection_id, LLMConnection.user_id == user_id, LLMConnection.deleted_at.is_(None)).with_for_update().execution_options(populate_existing=True))
        if not row or row.revision != revision:
            db.rollback()
            raise ModelSetupError("connection_changed_during_test")
        row.last_test_status = "failed" if failure else "passed"
        row.last_test_error = failure or ""
        row.last_tested_at = datetime.now(UTC).replace(tzinfo=None)
        if not failure:
            row.status = "active"
            tested_model = _add_model_row(db, row, {"model_id": selected_model, "source": "manual"})
            _ensure_default(db, user_id, row, tested_model.id)
        db.commit()
    if failure:
        raise ModelSetupError(failure)
    return {"status": "ok", "model_id": selected_model, "models": [], "persisted": persist,
            "latency_ms": round((perf_counter() - started) * 1000)}


def discover_models(db: Session, user_id: int, connection_id: int) -> list[dict[str, Any]]:
    row = _connection(db, user_id, connection_id)
    fields = {**(row.config_json or {}), **decrypt_secrets(row.encrypted_secrets)}
    kind = _discovery_kind(row.provider, row.protocol)
    if kind == "none":
        raise ModelSetupError("model_discovery_unavailable")
    revision = row.revision
    db.rollback()
    try:
        models = _discover(kind, fields)
    except Exception as exc:
        raise ModelSetupError(_classify_error(exc)) from exc
    row = db.scalar(select(LLMConnection).where(LLMConnection.id == connection_id, LLMConnection.user_id == user_id, LLMConnection.deleted_at.is_(None)).with_for_update().execution_options(populate_existing=True))
    if not row or row.revision != revision:
        db.rollback()
        raise ModelSetupError("connection_changed_during_test")
    _upsert_discovered_models(db, row, models)
    db.commit()
    return [_model_response(item) for item in db.scalars(select(LLMModel).where(LLMModel.connection_id == row.id, LLMModel.enabled.is_(True))).all()]


def get_preferences(db: Session, user_id: int) -> dict[str, Any]:
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    return {"default_model_config_id": profile.default_llm_model_id if profile else None}


def update_preferences(db: Session, user_id: int, model_id: int | None) -> dict[str, Any]:
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
    if model_id is not None:
        resolve_model_context(db, user_id, model_id)
    profile.default_llm_model_id = model_id
    db.commit()
    return {"default_model_config_id": model_id}


def resolve_model_context(db: Session, user_id: int, model_config_id: int | None, conversation_id: str | None = None) -> ModelExecutionContext:
    if model_config_id is None and conversation_id:
        conversation = db.scalar(select(AgentConversation).where(AgentConversation.user_id == user_id, AgentConversation.conversation_id == conversation_id))
        if conversation:
            model_config_id = (conversation.metadata_json or {}).get("model_config_id")
    if model_config_id is None:
        profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
        model_config_id = profile.default_llm_model_id if profile else None
    if not model_config_id:
        raise ModelSetupError("model_setup_required")
    row = db.execute(select(LLMModel, LLMConnection).join(LLMConnection, LLMConnection.id == LLMModel.connection_id).where(LLMModel.id == int(model_config_id), LLMConnection.user_id == user_id, LLMConnection.deleted_at.is_(None))).first()
    if not row:
        raise ModelSetupError("model_not_found")
    model, connection = row
    if not model.enabled or connection.status != "active" or connection.last_test_status != "passed":
        raise ModelSetupError("model_not_available")
    capabilities = {**DEFAULT_CAPABILITIES, **(model.capabilities_json or {})}
    missing = [key for key in ("tools", "structured_output", "streaming") if not capabilities.get(key)]
    if missing:
        raise ModelSetupError(f"model_missing_capabilities:{','.join(missing)}")
    return ModelExecutionContext(
        model_config_id=model.id,
        connection_id=connection.id,
        connection_revision=connection.revision,
        provider=connection.provider,
        protocol=connection.protocol,
        model=model.model_id,
        display_name=model.display_name,
        config=dict(connection.config_json or {}),
        secrets=decrypt_secrets(connection.encrypted_secrets),
        capabilities=capabilities,
    )


def resolve_run_model_context(db: Session, user_id: int, snapshot: dict[str, Any]) -> ModelExecutionContext:
    """Restore an existing Run's immutable routing choice with late-bound credentials.

    A connection may be edited or soft-deleted after a Run starts. Existing Runs keep
    their original provider/protocol/model selection, while credentials and endpoint
    configuration are decrypted only when the Run executes or resumes.
    """
    try:
        model_config_id = int(snapshot["model_config_id"])
        connection_id = int(snapshot["connection_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelSetupError("run_model_context_missing") from exc
    row = db.execute(
        select(LLMModel, LLMConnection)
        .join(LLMConnection, LLMConnection.id == LLMModel.connection_id)
        .where(
            LLMModel.id == model_config_id,
            LLMConnection.id == connection_id,
            LLMConnection.user_id == user_id,
        )
    ).first()
    if not row:
        raise ModelSetupError("run_model_connection_not_found")
    model, connection = row
    capabilities = {**DEFAULT_CAPABILITIES, **dict(snapshot.get("capabilities") or {})}
    return ModelExecutionContext(
        model_config_id=model_config_id,
        connection_id=connection_id,
        connection_revision=int(snapshot.get("connection_revision") or connection.revision),
        provider=str(snapshot.get("provider") or connection.provider),
        protocol=str(snapshot.get("protocol") or connection.protocol),
        model=str(snapshot.get("model") or model.model_id),
        display_name=str(snapshot.get("display_name") or model.display_name),
        config=dict(connection.config_json or {}),
        secrets=decrypt_secrets(connection.encrypted_secrets),
        capabilities=capabilities,
    )


def remember_conversation_model(db: Session, user_id: int, conversation_id: str, model_config_id: int) -> None:
    conversation = db.scalar(select(AgentConversation).where(AgentConversation.user_id == user_id, AgentConversation.conversation_id == conversation_id))
    if conversation:
        conversation.metadata_json = {**(conversation.metadata_json or {}), "model_config_id": model_config_id}


def _connection(db: Session, user_id: int, connection_id: int) -> LLMConnection:
    row = db.scalar(select(LLMConnection).where(LLMConnection.id == connection_id, LLMConnection.user_id == user_id, LLMConnection.deleted_at.is_(None)))
    if not row:
        raise ModelSetupError("connection_not_found")
    return row


def _connection_response(db: Session, row: LLMConnection) -> dict[str, Any]:
    secrets = decrypt_secrets(row.encrypted_secrets)
    models = db.scalars(select(LLMModel).where(LLMModel.connection_id == row.id).order_by(LLMModel.display_name)).all()
    return {
        "id": row.id, "provider": row.provider, "protocol": row.protocol, "display_name": row.display_name,
        "fields": row.config_json or {}, "secrets": {key: {"configured": bool(value), "masked": "••••" if isinstance(value, dict) else masked_secret(str(value))} for key, value in secrets.items()},
        "revision": row.revision, "status": row.status, "last_test_status": row.last_test_status,
        "last_test_error": row.last_test_error, "last_tested_at": row.last_tested_at.isoformat() if row.last_tested_at else None,
        "models": [_model_response(model) for model in models],
    }


def _model_response(model: LLMModel) -> dict[str, Any]:
    return {"id": model.id, "connection_id": model.connection_id, "model_id": model.model_id, "display_name": model.display_name, "source": model.source, "capabilities": model.capabilities_json or {}, "enabled": model.enabled}


def _add_model_row(db: Session, connection: LLMConnection, payload: dict[str, Any]) -> LLMModel:
    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        raise ModelSetupError("model_id_required")
    source = str(payload.get("source") or "manual")
    if source not in {"preset", "discovered", "manual"}:
        raise ModelSetupError("invalid_model_source")
    existing = db.scalar(select(LLMModel).where(LLMModel.connection_id == connection.id, LLMModel.model_id == model_id))
    if existing:
        existing.enabled = True
        return existing
    model = LLMModel(connection_id=connection.id, model_id=model_id, display_name=str(payload.get("display_name") or model_id), source=source, capabilities_json={**DEFAULT_CAPABILITIES, **dict(payload.get("capabilities") or {})}, enabled=True)
    db.add(model)
    db.flush()
    return model


def _apply_field_defaults(provider: Any, fields: dict[str, Any]) -> None:
    for definition in provider.fields:
        if definition.key not in fields and definition.default not in (None, ""):
            fields[definition.key] = definition.default
    if provider.key == "qwen" and fields.get("region") == "intl" and fields.get("base_url") == "https://dashscope.aliyuncs.com/compatible-mode/v1":
        fields["base_url"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _validate_fields(provider: Any, fields: dict[str, Any]) -> None:
    allowed = {definition.key for definition in provider.fields}
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ModelSetupError(f"unsupported_fields:{','.join(unknown)}")
    missing = [definition.key for definition in provider.fields if definition.required and fields.get(definition.key) in (None, "")]
    if missing:
        raise ModelSetupError(f"missing_required_fields:{','.join(missing)}")
    if "custom_headers" in fields and not isinstance(fields["custom_headers"], dict):
        raise ModelSetupError("custom_headers_must_be_an_object")
    if any(not isinstance(k, str) or not isinstance(v, str) or any(c in k + v for c in "\r\n") for k, v in (fields.get("custom_headers") or {}).items()):
        raise ModelSetupError("invalid_custom_headers")
    for definition in provider.fields:
        if definition.kind == "url" and fields.get(definition.key):
            parsed = urlsplit(str(fields[definition.key]).strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ModelSetupError(f"invalid_url:{definition.key}")
            fields[definition.key] = str(fields[definition.key]).strip().rstrip("/")


def _discovery_kind(provider_key: str, protocol: str) -> str:
    if provider_key != "custom":
        return get_provider(provider_key).discovery
    return {"openai_chat_completions": "openai_models", "openai_responses": "openai_models",
            "anthropic_messages": "anthropic_models", "ollama_chat": "ollama_tags"}.get(protocol, "none")


def _split_fields(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return ({key: value for key, value in fields.items() if key in SECRET_FIELD_KEYS and value not in (None, "")}, {key: value for key, value in fields.items() if key not in SECRET_FIELD_KEYS})


def _discover(kind: str, fields: dict[str, Any]) -> list[dict[str, Any]]:
    timeout = float(get_settings().llm_timeout_seconds)
    api_key = str(fields.get("api_key") or "")
    protocol = "anthropic_messages" if kind == "anthropic_models" else "ollama_chat" if kind == "ollama_tags" else "openai_chat_completions"
    base_url = normalize_model_endpoint(str(fields.get("base_url") or ""), protocol)
    if kind == "none":
        return []
    if kind == "ollama_tags":
        response = httpx.get(f"{base_url.removesuffix('/v1')}/api/tags", headers=dict(fields.get("custom_headers") or {}), timeout=timeout)
        response.raise_for_status()
        return [{"model_id": item.get("model") or item.get("name"), "display_name": item.get("name") or item.get("model")} for item in response.json().get("models", []) if item.get("model") or item.get("name")]
    if kind == "gemini_models":
        items: list[dict[str, Any]] = []
        page_token = ""
        for _ in range(20):
            params = {"key": api_key, **({"pageToken": page_token} if page_token else {})}
            response = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            items.extend(payload.get("models", []))
            page_token = str(payload.get("nextPageToken") or "")
            if not page_token:
                break
        return [{"model_id": str(item.get("name", "")).removeprefix("models/"), "display_name": item.get("displayName") or item.get("name")} for item in items if "generateContent" in item.get("supportedGenerationMethods", [])]
    headers = {"Authorization": f"Bearer {api_key}"}
    if kind == "anthropic_models":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        url = f"{base_url.removesuffix('/v1')}/v1/models"
    else:
        url = f"{base_url}/models"
    if fields.get("auth_header") not in (None, "", "Authorization"):
        headers.pop("Authorization", None)
        if api_key:
            headers[str(fields["auth_header"])] = api_key
    headers.update(dict(fields.get("custom_headers") or {}))
    items = []
    after = ""
    for _ in range(20):
        response = httpx.get(url, headers=headers, params={"after": after} if after else None, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("data", [])
        items.extend(page)
        if not payload.get("has_more") or not page:
            break
        after = str(payload.get("last_id") or page[-1].get("id") or "")
        if not after:
            break
    return [{"model_id": item.get("id"), "display_name": item.get("display_name") or item.get("name") or item.get("id")} for item in items if item.get("id")]


def _upsert_discovered_models(db: Session, connection: LLMConnection, models: list[dict[str, Any]]) -> None:
    for model in models:
        # Refresh must not silently re-enable models explicitly disabled by the user.
        existing = db.scalar(select(LLMModel.id).where(LLMModel.connection_id == connection.id, LLMModel.model_id == model["model_id"]))
        if not existing:
            _add_model_row(db, connection, {**model, "source": "discovered"})


def _ensure_default(db: Session, user_id: int, connection: LLMConnection, tested_model_id: int | None = None) -> None:
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
        db.flush()
    if profile.default_llm_model_id is None:
        if tested_model_id:
            profile.default_llm_model_id = tested_model_id
            return
        first_model = db.scalar(select(LLMModel).where(LLMModel.connection_id == connection.id, LLMModel.enabled.is_(True)).order_by(LLMModel.id))
        if first_model:
            profile.default_llm_model_id = first_model.id


def _classify_error(exc: Exception) -> str:
    if isinstance(exc, ModelSetupError):
        return str(exc)
    if isinstance(exc, httpx.ConnectError):
        return "connection_unreachable"
    if isinstance(exc, httpx.TimeoutException):
        return "connection_timeout"
    if isinstance(exc, httpx.HTTPStatusError) or getattr(exc, "status_code", None):
        code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else exc.status_code
        if code in (401, 403): return "authentication_failed"
        if code == 404: return "endpoint_or_model_not_found"
        if code == 429: return "rate_limited"
        return f"provider_http_error:{code}"
    if "timeout" in type(exc).__name__.lower():
        return "connection_timeout"
    if type(exc).__name__ in {"APIConnectionError", "ConnectError"}:
        return "connection_unreachable"
    return f"connection_failed:{type(exc).__name__}"
