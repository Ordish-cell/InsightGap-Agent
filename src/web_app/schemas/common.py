from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: ErrorBody | None = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))


def ok(data: Any = None) -> dict[str, Any]:
    return ApiResponse(success=True, data=data).model_dump()


def fail(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return ApiResponse(success=False, error=ErrorBody(code=code, message=message, details=details or {})).model_dump()
