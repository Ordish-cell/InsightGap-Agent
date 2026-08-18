from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from src.web_app.agent.llm.errors import LLMUnavailableError


@dataclass(frozen=True)
class ModelExecutionContext:
    model_config_id: int
    connection_id: int
    connection_revision: int
    provider: str
    protocol: str
    model: str
    display_name: str
    config: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, Any] = field(default_factory=dict, repr=False)
    capabilities: dict[str, bool] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("secrets", None)
        payload.pop("config", None)
        return payload


_current_model_context: ContextVar[ModelExecutionContext | None] = ContextVar("current_model_context", default=None)


def get_model_context() -> ModelExecutionContext:
    context = _current_model_context.get()
    if context is None:
        raise LLMUnavailableError("model_setup_required")
    return context


@contextmanager
def use_model_context(context: ModelExecutionContext) -> Iterator[None]:
    token = _current_model_context.set(context)
    try:
        yield
    finally:
        _current_model_context.reset(token)


def activate_model_context(context: ModelExecutionContext) -> Token:
    return _current_model_context.set(context)


def reset_model_context(token: Token) -> None:
    _current_model_context.reset(token)
