from typing import Any


class DocumentIngestError(Exception):
    """Raised when document ingestion fails and the caller should return non-200."""

    def __init__(self, message: str, *, document_id: int | None = None, http_status: int = 500, detail: str = "") -> None:
        super().__init__(message)
        self.document_id = document_id
        self.http_status = http_status
        self.detail = detail or message


def error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details or {}}
