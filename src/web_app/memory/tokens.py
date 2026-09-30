"""Token budgets preserve both recent corrections and older task context."""
from functools import lru_cache


@lru_cache(maxsize=1)
def encoding():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count(text):
    return len(encoding().encode(text, disallowed_special=()))


def fit(text, budget):
    if budget <= 0:
        return ""
    tokens = encoding().encode(text, disallowed_special=())
    if len(tokens) <= budget:
        return text
    marker = "\n[中间内容省略；可检索原始消息]\n"
    if budget <= count(marker):
        return encoding().decode(tokens[-budget:], errors="ignore")
    available = max(0, budget - count(marker))
    head = available // 3
    tail = available - head
    return encoding().decode(tokens[:head], errors="ignore") + marker + (encoding().decode(tokens[-tail:], errors="ignore") if tail else "")


def chunks(text, budget=5000):
    tokens = encoding().encode(text, disallowed_special=())
    if budget < 4:
        raise ValueError("chunk budget must allow a complete UTF-8 character")
    parts, start = [], 0
    while start < len(tokens):
        end = min(start + budget, len(tokens))
        while end > start:
            try:
                part = encoding().decode_bytes(tokens[start:end]).decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
        parts.append(part)
        start = end
    return parts or [""]
