from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.web_app.agent.llm.context import ModelExecutionContext
from src.web_app.agent.llm.crypto import decrypt_secrets, encrypt_secrets, masked_secret
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
    if provider.key == "custom" and not model_id:
        raise ModelSetupError("model_id_required")
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
    _validate_fields(provider, {**fields, **old_secrets, **incoming_secrets})
    row.config_json = fields
    row.encrypted_secrets = encrypt_secrets({**old_secrets, **incoming_secrets})
    if payload.get("display_name") is not None:
        row.display_name = str(payload["display_name"]).strip()
    if payload.get("protocol") is not None:
        protocol = str(payload["protocol"])
        if protocol not in (provider.protocols or (provider.protocol,)):
            raise ModelSetupError("protocol_not_supported")
        row.protocol = protocol
    row.revision += 1
    row.status = "draft"
    row.last_test_status = "untested"
    row.last_test_error = ""
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
    connection_id = payload.get("connection_id")
    if connection_id:
        row = _connection(db, user_id, int(connection_id))
        provider_key, protocol = row.provider, row.protocol
        fields = {**(row.config_json or {}), **decrypt_secrets(row.encrypted_secrets)}
        supplied = dict(payload.get("fields") or {})
        fields.update({key: value for key, value in supplied.items() if value not in (None, "")})
    else:
        row = None
        provider_key = str(payload.get("provider") or "")
        provider = get_provider(provider_key)
        protocol = str(payload.get("protocol") or provider.protocol)
        fields = dict(payload.get("fields") or {})
        _apply_field_defaults(provider, fields)
    provider = get_provider(provider_key)
    if protocol not in (provider.protocols or (provider.protocol,)):
        raise ModelSetupError("protocol_not_supported")
    _validate_fields(provider, fields)
    try:
        models = _discover(provider.discovery, fields)
        if provider.discovery == "none":
            selected_model = str(payload.get("model_id") or fields.get("deployment") or "").strip()
            if not selected_model:
                raise ModelSetupError("model_id_required")
            from src.web_app.agent.llm.factory import build_chat_model
            temporary = ModelExecutionContext(0, 0, 0, provider_key, protocol, selected_model, "Connection test", {key: value for key, value in fields.items() if key not in SECRET_FIELD_KEYS}, {key: value for key, value in fields.items() if key in SECRET_FIELD_KEYS}, dict(DEFAULT_CAPABILITIES))
            build_chat_model(temporary, temperature=None).invoke("Reply with OK")
        result = {"status": "ok", "models": models}
        if row:
            row.status = "active"
            row.last_test_status = "passed"
            row.last_test_error = ""
            row.last_tested_at = datetime.now(UTC).replace(tzinfo=None)
            _upsert_discovered_models(db, row, models)
            _ensure_default(db, user_id, row)
            db.commit()
        return result
    except Exception as exc:
        if row:
            row.status = "draft"
            row.last_test_status = "failed"
            row.last_test_error = _safe_error(exc, [str(fields.get(key) or "") for key in SECRET_FIELD_KEYS])
            row.last_tested_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()
        raise ModelSetupError(_classify_error(exc)) from exc


def discover_models(db: Session, user_id: int, connection_id: int) -> list[dict[str, Any]]:
    row = _connection(db, user_id, connection_id)
    if row.status != "active":
        raise ModelSetupError("connection_not_verified")
    fields = {**(row.config_json or {}), **decrypt_secrets(row.encrypted_secrets)}
    models = _discover(get_provider(row.provider).discovery, fields)
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


def _split_fields(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return ({key: value for key, value in fields.items() if key in SECRET_FIELD_KEYS and value not in (None, "")}, {key: value for key, value in fields.items() if key not in SECRET_FIELD_KEYS})


def _discover(kind: str, fields: dict[str, Any]) -> list[dict[str, Any]]:
    timeout = float(get_settings().llm_timeout_seconds)
    api_key = str(fields.get("api_key") or "")
    base_url = str(fields.get("base_url") or "").rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if base_url.endswith(suffix):
            base_url = base_url[:-len(suffix)]
    if kind == "none":
        return []
    if kind == "ollama_tags":
        response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
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
        url = f"{base_url}/v1/models"
    else:
        url = f"{base_url}/models"
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
        _add_model_row(db, connection, {**model, "source": "discovered"})


def _ensure_default(db: Session, user_id: int, connection: LLMConnection) -> None:
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
        db.flush()
    if profile.default_llm_model_id is None:
        first_model = db.scalar(select(LLMModel).where(LLMModel.connection_id == connection.id, LLMModel.enabled.is_(True)).order_by(LLMModel.id))
        if first_model:
            profile.default_llm_model_id = first_model.id


def _safe_error(exc: Exception, secret_values: list[str] | None = None) -> str:
    message = str(exc).replace("\n", " ")
    for value in secret_values or []:
        if value:
            message = message.replace(value, "***")
    return message[:500]


def _classify_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "connection_timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403): return "authentication_failed"
        if code == 404: return "endpoint_or_model_not_found"
        if code == 429: return "rate_limited"
        return f"provider_http_error:{code}"
    return f"connection_failed:{type(exc).__name__}"
