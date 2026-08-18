from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConnectionCreate(BaseModel):
    provider: str
    protocol: str | None = None
    display_name: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    model_id: str = ""
    model_display_name: str = ""
    models: list[dict[str, Any]] = Field(default_factory=list)


class ConnectionUpdate(BaseModel):
    protocol: str | None = None
    display_name: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)


class ConnectionTest(BaseModel):
    connection_id: int | None = None
    provider: str = ""
    protocol: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    model_id: str = ""


class ModelCreate(BaseModel):
    model_id: str
    display_name: str = ""
    source: str = "manual"
    capabilities: dict[str, bool] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    model_id: str | None = None
    display_name: str | None = None
    source: str | None = None
    enabled: bool | None = None
    capabilities: dict[str, bool] | None = None


class PreferenceUpdate(BaseModel):
    default_model_config_id: int | None = None
