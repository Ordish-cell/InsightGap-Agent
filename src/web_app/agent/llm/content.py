"""Extract answer text from model messages without serializing reasoning blocks."""
from typing import Any


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(message_text(block) for block in content)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {None, "text", "output_text"}:
            text = content.get("text", "")
            return text if isinstance(text, str) else ""
        if kind == "message":
            return message_text(content.get("content", []))
    return ""
