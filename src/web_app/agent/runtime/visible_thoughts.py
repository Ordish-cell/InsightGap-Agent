"""Read-only projection of visible thoughts stored by historical runs."""
from typing import Any


def visible_thought_texts(state: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in state.get("visible_thoughts") or []:
        text = item.get("text") if isinstance(item, dict) else item
        text = str(text or "").strip()
        if text and text not in texts:
            texts.append(text)
    return texts
