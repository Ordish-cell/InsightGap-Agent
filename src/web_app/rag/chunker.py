from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_index: int
    content: str
    token_count: int
    char_start: int
    char_end: int
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "content": self.content,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "heading_path": self.heading_path,
            "metadata": self.metadata,
        }


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def chunk_markdown(markdown: str, chunk_size: int = 1000, overlap: int = 120) -> list[dict[str, Any]]:
    if not markdown or not markdown.strip():
        return []
    blocks = _split_blocks(markdown)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_start = 0
    heading_path: list[str] = []
    active_heading: list[str] = []

    for block, start, end, block_heading in blocks:
        if block_heading is not None:
            level, title = block_heading
            active_heading = active_heading[: level - 1] + [title]
        candidate = "\n\n".join(current + [block]).strip()
        if current and estimate_tokens(candidate) > chunk_size:
            _append_chunk(chunks, "\n\n".join(current), current_start, start, heading_path)
            tail = _overlap_tail(current, overlap)
            current = tail + [block]
            current_start = max(0, start - sum(len(item) for item in tail))
        else:
            if not current:
                current_start = start
            current.append(block)
        heading_path = list(active_heading)

    if current:
        _append_chunk(chunks, "\n\n".join(current), current_start, len(markdown), heading_path)
    return [chunk.as_dict() for chunk in chunks if chunk.content.strip()]


def _split_blocks(markdown: str) -> list[tuple[str, int, int, tuple[int, str] | None]]:
    blocks: list[tuple[str, int, int, tuple[int, str] | None]] = []
    cursor = 0
    for raw in markdown.split("\n\n"):
        block = raw.strip()
        start = markdown.find(raw, cursor)
        end = start + len(raw)
        cursor = end
        if not block:
            continue
        heading = None
        first = block.splitlines()[0].strip()
        if first.startswith("#"):
            marks = len(first) - len(first.lstrip("#"))
            if 1 <= marks <= 6:
                heading = (marks, first.lstrip("#").strip())
        blocks.append((block, max(start, 0), max(end, 0), heading))
    return blocks


def _append_chunk(chunks: list[Chunk], content: str, start: int, end: int, heading_path: list[str]) -> None:
    content = content.strip()
    if not content:
        return
    if estimate_tokens(content) > 1400:
        _split_large_chunk(chunks, content, start, heading_path)
        return
    chunks.append(Chunk(len(chunks), content, estimate_tokens(content), start, end, list(heading_path), {}))


def _split_large_chunk(chunks: list[Chunk], content: str, start: int, heading_path: list[str], max_chars: int = 4000, overlap_chars: int = 500) -> None:
    cursor = 0
    while cursor < len(content):
        part = content[cursor : cursor + max_chars].strip()
        if part:
            chunks.append(Chunk(len(chunks), part, estimate_tokens(part), start + cursor, start + cursor + len(part), list(heading_path), {}))
        if cursor + max_chars >= len(content):
            break
        cursor += max_chars - overlap_chars


def _overlap_tail(blocks: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    used = 0
    for block in reversed(blocks):
        tokens = estimate_tokens(block)
        if used + tokens > overlap:
            break
        tail.insert(0, block)
        used += tokens
    return tail
