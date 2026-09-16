from __future__ import annotations

import logging
from typing import Any

from src.web_app.rag.chunker import estimate_tokens

logger = logging.getLogger(__name__)

SUMMARY_INPUT_CHARS = 16_000
EXTRACTIVE_SUMMARY_CHARS = 4_000


def build_document_summary(
    filename: str,
    parent_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = [_source_from_parent(chunk) for chunk in parent_chunks if str(chunk.get("content") or "").strip()]
    if not sources:
        return {"status": "failed", "summary_text": "", "section_summaries": [], "error": "No parent content"}

    total_chars = sum(len(item["content"]) for item in sources)
    if total_chars <= EXTRACTIVE_SUMMARY_CHARS:
        text = "\n\n".join(item["content"] for item in sources).strip()
        return {
            "status": "generated",
            "method": "extractive_full_text",
            "summary_text": text,
            "section_summaries": [_section_result(item, item["content"]) for item in sources],
            "error": "",
        }

    try:
        section_summaries = []
        for group in _group_sources(sources, SUMMARY_INPUT_CHARS):
            summary = _llm_summary(
                "请忠实概括以下文档片段，保留核心观点、结论、数据和风险，不要补充原文没有的信息。"
                "输出简洁中文正文，不要使用代码块。\n\n" + _format_sources(group)
            )
            section_summaries.append(_section_result(group[0], summary, group[-1]))

        reduce_inputs = [item["summary_text"] for item in section_summaries]
        while len("\n\n".join(reduce_inputs)) > SUMMARY_INPUT_CHARS:
            reduced: list[str] = []
            for group in _group_texts(reduce_inputs, SUMMARY_INPUT_CHARS):
                reduced.append(_llm_summary(
                    "请合并以下分段摘要，去重并保留整份文档的重要主题、结论、数据和风险。"
                    "不要编造。输出简洁中文正文。\n\n" + "\n\n".join(group)
                ))
            reduce_inputs = reduced
        final_summary = _llm_summary(
            f"请根据以下分段摘要，生成《{filename}》的全文概括。覆盖文档的不同部分，"
            "不要只描述开头，不要编造。输出1000字以内的中文正文。\n\n" + "\n\n".join(reduce_inputs)
        )
        return {
            "status": "generated",
            "method": "hierarchical_llm_v1",
            "summary_text": final_summary,
            "section_summaries": section_summaries,
            "error": "",
        }
    except Exception as exc:
        logger.warning("document.summary_failed filename=%s error=%s", filename, exc, exc_info=True)
        return {"status": "failed", "summary_text": "", "section_summaries": [], "error": str(exc)}


def summary_chunks(summary: dict[str, Any], start_index: int) -> list[dict[str, Any]]:
    from src.web_app.rag.embeddings import MAX_EMBED_CHARS
    chunks: list[dict[str, Any]] = []
    def append_parts(content, role, identifier, metadata):
        parts = [content[start:start + MAX_EMBED_CHARS] for start in range(0, len(content), MAX_EMBED_CHARS)]
        for number, part in enumerate(parts, 1):
            chunk_id = identifier if len(parts) == 1 else f'{identifier}-part-{number:04d}'
            chunks.append(_chunk(start_index + len(chunks), part, role, chunk_id,
                {**metadata, 'summary_part': number, 'summary_part_count': len(parts)}))
    overview = str(summary.get("summary_text") or "").strip()
    if overview:
        append_parts(overview, "overview", "overview-0000", summary)
    for number, item in enumerate(summary.get("section_summaries") or [], 1):
        content = str(item.get("summary_text") or "").strip()
        if not content:
            continue
        metadata = {
            "parent_ids": item.get("parent_ids", []),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "heading_path": item.get("heading_path", []),
        }
        append_parts(content, "section_summary", f"summary-{number:04d}", metadata)
    return chunks


def _chunk(index: int, content: str, role: str, chunk_id: str, extra: dict[str, Any]) -> dict[str, Any]:
    import hashlib

    metadata = {
        "chunk_role": role,
        "chunk_type": role,
        "chunk_id": chunk_id,
        "parent_id": None,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        **{key: value for key, value in extra.items() if key in {"parent_ids", "page_start", "page_end", "heading_path", "summary_part", "summary_part_count"}},
    }
    return {
        "chunk_index": index,
        "content": content,
        "token_count": estimate_tokens(content),
        "char_start": 0,
        "char_end": len(content),
        "heading_path": metadata.get("heading_path", []),
        "metadata": metadata,
    }


def _source_from_parent(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata", {}) or {}
    return {
        "content": str(chunk.get("content") or ""),
        "parent_id": str(metadata.get("chunk_id") or ""),
        "page_number": metadata.get("page_number"),
        "heading_path": metadata.get("heading_path", []),
    }


def _section_result(first: dict[str, Any], summary: str, last: dict[str, Any] | None = None) -> dict[str, Any]:
    last = last or first
    return {
        "summary_text": summary.strip(),
        "parent_ids": [item for item in [first.get("parent_id"), last.get("parent_id")] if item],
        "page_start": first.get("page_number"),
        "page_end": last.get("page_number"),
        "heading_path": first.get("heading_path", []),
    }


def _group_sources(sources: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for source in sources:
        content = source["content"]
        if current and size + len(content) > max_chars:
            groups.append(current)
            current = []
            size = 0
        if len(content) > max_chars:
            for start in range(0, len(content), max_chars):
                part = {**source, "content": content[start:start + max_chars]}
                if current:
                    groups.append(current)
                    current = []
                    size = 0
                groups.append([part])
            continue
        current.append(source)
        size += len(content)
    if current:
        groups.append(current)
    return groups


def _group_texts(texts: list[str], max_chars: int) -> list[list[str]]:
    sources = [{"content": text} for text in texts]
    return [[item["content"] for item in group] for group in _group_sources(sources, max_chars)]


def _format_sources(sources: list[dict[str, Any]]) -> str:
    parts = []
    for item in sources:
        label = item.get("parent_id") or "section"
        if item.get("page_number"):
            label += f" page={item['page_number']}"
        parts.append(f"[{label}]\n{item['content']}")
    return "\n\n".join(parts)


def _llm_summary(prompt: str) -> str:
    from src.web_app.agent.llm.factory import get_chat_model
    from src.web_app.agent.llm.content import message_text

    message = get_chat_model("rag", complexity="low", temperature=0.1).invoke(prompt)
    result = message_text(message).strip()
    if not result:
        raise RuntimeError("Summary model returned empty content")
    return result
