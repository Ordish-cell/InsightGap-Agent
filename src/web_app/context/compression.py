def compress_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 16] + "\n[compressed]"
