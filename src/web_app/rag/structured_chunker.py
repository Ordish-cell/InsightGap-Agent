from __future__ import annotations

import hashlib
import csv
import re
from dataclasses import dataclass, field
from typing import Any

from src.web_app.rag.chunker import chunk_markdown, estimate_tokens

MAX_CHILD_CHARS = 7000
CHILD_CHAR_OVERLAP = 400


@dataclass
class StructuredChunk:
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


def build_structured_chunks(
    text: str,
    *,
    file_type: str,
    filename: str,
    parser_metadata: dict[str, Any] | None = None,
    child_chunk_size: int = 800,
    child_overlap: int = 80,
) -> dict[str, Any]:
    """Build overview, parent chunks, and searchable child chunks.

    The returned ``chunks`` list is intended for PostgreSQL. Only chunks with
    ``metadata.chunk_role == "child"`` should be embedded/upserted to Qdrant.
    """
    parser_metadata = parser_metadata or {}
    normalized_type = (file_type or "").lower().lstrip(".")
    content = text or ""
    if not content.strip():
        return {"chunks": [], "vector_chunks": [], "overview": {}, "document_map": {}, "stats": {}}

    if normalized_type in {"csv", "xlsx"}:
        parents = _table_parents(content, normalized_type, parser_metadata)
    else:
        parents = _textual_parents(content, normalized_type, parser_metadata)

    overview = _build_overview(filename, normalized_type, parser_metadata, parents)
    document_map = _build_document_map(filename, normalized_type, parser_metadata, parents)

    chunks: list[StructuredChunk] = []
    overview_chunk = _make_chunk(
        chunks,
        overview.get("summary_text", ""),
        0,
        0,
        [],
        {
            "chunk_role": "overview",
            "chunk_type": "overview",
            "chunk_id": "overview-0000",
            "parent_id": None,
            "content_hash": _hash(overview.get("summary_text", "")),
        },
    )
    if overview_chunk is not None:
        chunks.append(overview_chunk)

    vector_chunks: list[dict[str, Any]] = []
    for parent_no, parent in enumerate(parents, 1):
        parent_id = f"p-{parent_no:04d}"
        parent_metadata = {
            "chunk_role": "parent",
            "chunk_type": parent.get("chunk_type", "section"),
            "chunk_id": parent_id,
            "parent_id": None,
            "heading_path": parent.get("heading_path", []),
            "page_number": parent.get("page_number"),
            "sheet_name": parent.get("sheet_name"),
            "header": parent.get("header", []),
            "row_start": parent.get("row_start"),
            "row_end": parent.get("row_end"),
            "content_hash": _hash(parent["content"]),
        }
        parent_chunk = _make_chunk(
            chunks,
            parent["content"],
            parent.get("char_start", 0),
            parent.get("char_end", parent.get("char_start", 0) + len(parent["content"])),
            parent.get("heading_path", []),
            parent_metadata,
        )
        if parent_chunk is not None:
            chunks.append(parent_chunk)

        children = _children_for_parent(parent, parent_id, child_chunk_size, child_overlap)
        for child in children:
            child["chunk_index"] = len(chunks)
            child_chunk = StructuredChunk(**child)
            chunks.append(child_chunk)
            vector_chunks.append(child_chunk.as_dict())

    all_dicts = [chunk.as_dict() for chunk in chunks if chunk.content.strip()]
    child_count = len(vector_chunks)
    stats = {
        "parent_count": len(parents),
        "child_count": child_count,
        "overview_count": 1 if overview_chunk is not None else 0,
        "chunk_count": child_count,
        "chunking_strategy": f"structured_{normalized_type or 'text'}_v1",
    }
    return {
        "chunks": all_dicts,
        "vector_chunks": vector_chunks,
        "overview": overview,
        "document_map": document_map,
        "stats": stats,
    }


def fallback_structured_chunks(markdown: str) -> dict[str, Any]:
    chunks = chunk_markdown(markdown)
    vector_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        metadata.update({
            "chunk_role": "child",
            "chunk_type": "text",
            "chunk_id": f"c-{chunk['chunk_index']:04d}",
            "parent_id": None,
            "content_hash": _hash(chunk["content"]),
        })
        chunk["metadata"] = metadata
        vector_chunks.append(chunk)
    return {
        "chunks": chunks,
        "vector_chunks": vector_chunks,
        "overview": {},
        "document_map": {},
        "stats": {"parent_count": 0, "child_count": len(vector_chunks), "overview_count": 0, "chunk_count": len(vector_chunks), "chunking_strategy": "legacy_chunk_markdown"},
    }


def _textual_parents(text: str, file_type: str, parser_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    pages = parser_metadata.get("pages")
    if file_type == "pdf" and isinstance(pages, list) and pages:
        return _pdf_page_parents(pages)

    blocks = _markdown_blocks(text)
    parents: list[dict[str, Any]] = []
    current: list[str] = []
    current_start = 0
    active_heading: list[str] = []
    parent_heading: list[str] = []

    for block, start, end, heading in blocks:
        if heading:
            level, title = heading
            if current:
                parents.append(_parent("\n\n".join(current), current_start, start, parent_heading, "section"))
                current = []
            active_heading = active_heading[: level - 1] + [title]
            parent_heading = list(active_heading)
            current_start = start
            current.append(block)
            continue

        if not current:
            current_start = start
            parent_heading = list(active_heading)
        candidate = "\n\n".join(current + [block])
        if current and estimate_tokens(candidate) > 1400:
            parents.append(_parent("\n\n".join(current), current_start, start, parent_heading, _guess_chunk_type("\n\n".join(current))))
            current = [block]
            current_start = start
            parent_heading = list(active_heading)
        else:
            current.append(block)

    if current:
        parents.append(_parent("\n\n".join(current), current_start, len(text), parent_heading, _guess_chunk_type("\n\n".join(current))))
    return parents or [_parent(text, 0, len(text), [], "text")]


def _pdf_page_parents(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    cursor = 0
    for page in pages:
        page_text = str(page.get("text") or "").strip()
        if not page_text:
            continue
        page_number = page.get("page_number")
        content = f"# Page {page_number}\n\n{page_text}" if page_number else page_text
        parent = _parent(content, cursor, cursor + len(content), [], "page")
        parent["page_number"] = page_number
        parents.append(parent)
        cursor += len(content) + 2
    return parents


def _table_parents(text: str, file_type: str, parser_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    sheet_blocks = _split_sheets(text)
    parents: list[dict[str, Any]] = []
    cursor = 0
    for sheet_name, rows in sheet_blocks:
        if not rows:
            continue
        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        row_block_size = 30 if file_type == "xlsx" else 40
        if not data_rows:
            content = _format_table_block(sheet_name, header, [])
            parent = _parent(content, cursor, cursor + len(content), [sheet_name] if sheet_name else [], "table")
            parent.update({"sheet_name": sheet_name, "header": header, "row_start": 0, "row_end": 0})
            parents.append(parent)
            cursor += len(content) + 2
            continue
        for start in range(0, len(data_rows), row_block_size):
            block_rows = data_rows[start:start + row_block_size]
            content = _format_table_block(sheet_name, header, block_rows)
            parent = _parent(content, cursor, cursor + len(content), [sheet_name] if sheet_name else [], "row_block")
            parent.update({"sheet_name": sheet_name, "header": header, "row_start": start + 1, "row_end": start + len(block_rows)})
            parents.append(parent)
            cursor += len(content) + 2
    return parents or [_parent(text, 0, len(text), [], "table")]


def _children_for_parent(parent: dict[str, Any], parent_id: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    content = parent["content"]
    if parent.get("chunk_type") in {"table", "row_block"}:
        parts = [content]
    else:
        parts = [chunk["content"] for chunk in chunk_markdown(content, chunk_size=chunk_size, overlap=overlap)] or [content]
    parts = _split_oversized_child_parts(parts)

    children: list[dict[str, Any]] = []
    cursor = parent.get("char_start", 0)
    for child_no, part in enumerate(parts, 1):
        part = part.strip()
        if not part:
            continue
        chunk_id = f"{parent_id}-c-{child_no:03d}"
        metadata = {
            "chunk_role": "child",
            "chunk_type": parent.get("chunk_type", "text"),
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "heading_path": parent.get("heading_path", []),
            "page_number": parent.get("page_number"),
            "sheet_name": parent.get("sheet_name"),
            "header": parent.get("header", []),
            "row_start": parent.get("row_start"),
            "row_end": parent.get("row_end"),
            "content_hash": _hash(part),
        }
        children.append({
            "chunk_index": 0,
            "content": part,
            "token_count": estimate_tokens(part),
            "char_start": cursor,
            "char_end": cursor + len(part),
            "heading_path": parent.get("heading_path", []),
            "metadata": metadata,
        })
        cursor += max(1, len(part) - overlap)
    return children


def _split_oversized_child_parts(parts: list[str], max_chars: int = MAX_CHILD_CHARS, overlap_chars: int = CHILD_CHAR_OVERLAP) -> list[str]:
    safe_parts: list[str] = []
    for part in parts:
        text = (part or "").strip()
        if not text:
            continue
        if len(text) <= max_chars:
            safe_parts.append(text)
            continue
        cursor = 0
        step = max(1, max_chars - overlap_chars)
        while cursor < len(text):
            end = min(len(text), cursor + max_chars)
            split_at = _preferred_split_point(text, cursor, end)
            chunk = text[cursor:split_at].strip()
            if chunk:
                safe_parts.append(chunk)
            if split_at >= len(text):
                break
            cursor = max(0, split_at - overlap_chars)
            if cursor <= 0 or cursor >= split_at:
                cursor = min(len(text), split_at + step)
    return safe_parts


def _preferred_split_point(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)
    window_start = max(start + 1, end - 800)
    for separator in ("\n\n", "\n", "。", "；", ";", ". ", " "):
        index = text.rfind(separator, window_start, end)
        if index > start:
            return index + len(separator)
    return end


def _markdown_blocks(markdown: str) -> list[tuple[str, int, int, tuple[int, str] | None]]:
    blocks: list[tuple[str, int, int, tuple[int, str] | None]] = []
    cursor = 0
    in_code = False
    current: list[str] = []
    current_start = 0
    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        if not current:
            current_start = cursor
        current.append(line)
        if stripped.startswith("```"):
            in_code = not in_code
        boundary = (not in_code and not stripped)
        cursor += len(line)
        if boundary:
            _append_block(blocks, "".join(current), current_start, cursor)
            current = []
    if current:
        _append_block(blocks, "".join(current), current_start, len(markdown))
    return blocks


def _append_block(blocks: list[tuple[str, int, int, tuple[int, str] | None]], raw: str, start: int, end: int) -> None:
    block = raw.strip()
    if not block:
        return
    heading = None
    first = block.splitlines()[0].strip()
    match = re.match(r"^(#{1,6})\s+(.+)$", first)
    if match:
        heading = (len(match.group(1)), match.group(2).strip())
    blocks.append((block, start, end, heading))


def _split_sheets(text: str) -> list[tuple[str, list[list[str]]]]:
    sheets: list[tuple[str, list[list[str]]]] = []
    current_name = ""
    current_rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            if current_rows:
                sheets.append((current_name, current_rows))
            current_name = stripped[2:].strip()
            current_rows = []
            continue
        parsed_row = _parse_table_line(stripped)
        if parsed_row:
            current_rows.append(parsed_row)
    if current_rows:
        sheets.append((current_name, current_rows))
    return sheets or [("", [row for row in (_parse_table_line(line.strip()) for line in text.splitlines() if line.strip()) if row])]


def _parse_table_line(line: str) -> list[str]:
    if not line:
        return []
    if "|" in line:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(cell and set(cell) <= {"-", ":"} for cell in cells):
            return []
        return cells
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t")]
    if "," in line:
        try:
            return [cell.strip() for cell in next(csv.reader([line]))]
        except Exception:
            return [cell.strip() for cell in line.split(",")]
    return [line.strip()]


def _format_table_block(sheet_name: str, header: list[str], rows: list[list[str]]) -> str:
    lines: list[str] = []
    if sheet_name:
        lines.append(f"# {sheet_name}")
    if header:
        lines.append("Columns: " + " | ".join(header))
    for idx, row in enumerate(rows, 1):
        pairs = []
        for col_idx, value in enumerate(row):
            column = header[col_idx] if col_idx < len(header) and header[col_idx] else f"col_{col_idx + 1}"
            pairs.append(f"{column}: {value}")
        lines.append(f"Row {idx}: " + "; ".join(pairs))
    return "\n".join(lines)


def _build_overview(filename: str, file_type: str, parser_metadata: dict[str, Any], parents: list[dict[str, Any]]) -> dict[str, Any]:
    headings = []
    for parent in parents:
        for heading in parent.get("heading_path", []):
            if heading and heading not in headings:
                headings.append(heading)
    sheets = parser_metadata.get("sheet_names") or sorted({p.get("sheet_name") for p in parents if p.get("sheet_name")})
    summary_lines = [
        f"File: {filename}",
        f"Type: {file_type or 'unknown'}",
        f"Parser: {parser_metadata.get('parser', '')}",
        f"Parent blocks: {len(parents)}",
    ]
    if parser_metadata.get("page_count"):
        summary_lines.append(f"Pages: {parser_metadata.get('page_count')}")
    if sheets:
        summary_lines.append("Sheets: " + ", ".join(str(s) for s in sheets))
    if headings:
        summary_lines.append("Headings: " + " > ".join(headings[:12]))
    first_text = next((p["content"][:500].replace("\n", " ") for p in parents if p.get("content")), "")
    if first_text:
        summary_lines.append("Preview: " + first_text)
    return {
        "filename": filename,
        "file_type": file_type,
        "parser": parser_metadata.get("parser", ""),
        "page_count": parser_metadata.get("page_count", 0),
        "sheet_names": sheets,
        "headings": headings[:30],
        "summary_text": "\n".join(summary_lines),
    }


def _build_document_map(filename: str, file_type: str, parser_metadata: dict[str, Any], parents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "filename": filename,
        "file_type": file_type,
        "parser": parser_metadata.get("parser", ""),
        "page_count": parser_metadata.get("page_count", 0),
        "sheet_names": parser_metadata.get("sheet_names", []),
        "sections": [
            {
                "parent_id": f"p-{idx:04d}",
                "chunk_type": parent.get("chunk_type", "section"),
                "heading_path": parent.get("heading_path", []),
                "page_number": parent.get("page_number"),
                "sheet_name": parent.get("sheet_name"),
                "row_start": parent.get("row_start"),
                "row_end": parent.get("row_end"),
                "token_count": estimate_tokens(parent.get("content", "")),
            }
            for idx, parent in enumerate(parents, 1)
        ],
    }


def _parent(content: str, start: int, end: int, heading_path: list[str], chunk_type: str) -> dict[str, Any]:
    return {
        "content": content.strip(),
        "char_start": start,
        "char_end": end,
        "heading_path": heading_path,
        "chunk_type": chunk_type,
    }


def _make_chunk(
    chunks: list[StructuredChunk],
    content: str,
    start: int,
    end: int,
    heading_path: list[str],
    metadata: dict[str, Any],
) -> StructuredChunk | None:
    content = (content or "").strip()
    if not content:
        return None
    return StructuredChunk(len(chunks), content, estimate_tokens(content), start, end, heading_path, metadata)


def _guess_chunk_type(content: str) -> str:
    if "```" in content:
        return "code"
    if "\t" in content or "|" in content and "\n" in content:
        return "table"
    return "section"


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
