from __future__ import annotations

import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from src.web_app.core.config import get_settings


class CredentialEncryptionError(ValueError):
    pass


def _fernet() -> Fernet:
    value = get_settings().model_credentials_encryption_key.strip()
    if not value:
        raise CredentialEncryptionError("MODEL_CREDENTIALS_ENCRYPTION_KEY is required")
    try:
        return Fernet(value.encode("ascii"))
    except Exception as exc:
        raise CredentialEncryptionError("MODEL_CREDENTIALS_ENCRYPTION_KEY must be a valid Fernet key") from exc


def encrypt_secrets(values: dict[str, Any]) -> str:
    if not values:
        return ""
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _fernet().encrypt(payload).decode("ascii")


def decrypt_secrets(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = _fernet().decrypt(value.encode("ascii"))
        result = json.loads(payload.decode("utf-8"))
    except (InvalidToken, ValueError, json.JSONDecodeError) as exc:
        raise CredentialEncryptionError("Stored model credentials cannot be decrypted") from exc
    return result if isinstance(result, dict) else {}


def masked_secret(value: str) -> str:
    if not value:
        return ""
    return f"••••{value[-4:]}" if len(value) >= 4 else "••••"
