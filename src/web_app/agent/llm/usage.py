from typing import Any

from sqlalchemy.orm import Session

from src.web_app.agent.runtime.checkpoint import record_event
from src.web_app.core.config import settings
from src.web_app.db.repositories.agent_repository import LLMCallRepository


def record_llm_call(
    db: Session | None,
    *,
    run_id: int | None,
    thread_id: str = "",
    user_id: int | None = None,
    node_name: str,
    purpose: str,
    provider: str,
    model: str,
    tier: str,
    latency_ms: int,
    status: str,
    error_message: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    estimated_input_chars: int | None = None,
    estimated_output_chars: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not settings.agent_llm_usage_log_enabled or db is None:
        return
    safe_metadata = _safe_metadata(metadata or {})
    try:
        LLMCallRepository(db).create(
            run_id=run_id,
            thread_id=thread_id or "",
            user_id=user_id,
            node_name=node_name,
            purpose=purpose,
            provider=provider,
            model=model,
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_input_chars=estimated_input_chars,
            estimated_output_chars=estimated_output_chars,
            latency_ms=max(0, int(latency_ms)),
            status=status,
            error_message=(error_message or "")[:500],
            metadata_json=safe_metadata,
        )
        if run_id:
            record_event(
                db,
                run_id,
                "llm_call_failed" if status == "failed" else "llm_call_completed",
                {
                    "purpose": purpose,
                    "provider": provider,
                    "model": model,
                    "tier": tier,
                    "latency_ms": latency_ms,
                    "status": status,
                    "error_message": (error_message or "")[:200],
                },
                node_name=node_name,
                user_id=user_id,
                thread_id=thread_id,
            )
    except Exception:
        # Usage logging must never break an Agent run.
        return


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    blocked = {"api_key", "authorization", "prompt", "raw_prompt", "raw_output"}
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if key.lower() in blocked:
            continue
        if key == "input_preview" and isinstance(value, str):
            result[key] = value[:200]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result
