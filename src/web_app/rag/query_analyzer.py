from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

QueryType = Literal["summary", "exact", "table", "semantic", "document_reference", "general"]


@dataclass
class QueryAnalysis:
    query: str
    query_type: QueryType = "semantic"
    query_types: list[QueryType] = field(default_factory=list)
    exact_terms: list[str] = field(default_factory=list)
    is_summary: bool = False
    is_exact: bool = False
    is_table: bool = False
    is_document_reference: bool = False


SUMMARY_PATTERNS = [
    "总结", "概括", "讲什么", "主要内容", "overview", "summary", "summarize", "what is this document", "what is this file",
]
EXACT_HINT_PATTERNS = [
    "编号", "金额", "日期", "邮箱", "邮件", "手机号", "电话", "订单号", "合同号", "产品型号", "型号", "版本号",
    "url", "函数名", "配置", "key", "id", "code", "version", "email", "phone", "contract", "order",
]
TABLE_PATTERNS = ["表格", "sheet", "列", "字段", "哪一行", "row", "column", "header", "worksheet"]
SEMANTIC_PATTERNS = ["为什么", "如何", "怎么", "风险", "原因", "影响", "原理", "建议", "why", "how", "risk", "impact"]
DOCUMENT_REFERENCE_PATTERNS = ["这个文件", "刚才上传", "上一个pdf", "当前文档", "这份表格", "这个文档", "这份材料", "this document", "this file"]

EXACT_TOKEN_RE = re.compile(
    r"https?://[^\s)）]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|"
    r"\b[A-Za-z]{1,12}[-_/][A-Za-z0-9][A-Za-z0-9._/-]*\b|"
    r"\b[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+){1,}\b|"
    r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?\b|"
    r"\b1[3-9]\d{9}\b|"
    r"\b\d+(?:\.\d+)?\b"
)


def analyze_query(query: str) -> QueryAnalysis:
    try:
        return _analyze_query(query)
    except Exception:
        return QueryAnalysis(query=query or "", query_type="semantic", query_types=["semantic"])


def _analyze_query(query: str) -> QueryAnalysis:
    text = (query or "").strip()
    lower = text.lower()
    query_types: list[QueryType] = []

    is_summary = _contains_any(lower, SUMMARY_PATTERNS)
    is_exact = _contains_any(lower, EXACT_HINT_PATTERNS)
    is_table = _contains_any(lower, TABLE_PATTERNS)
    is_document_reference = _contains_any(lower, DOCUMENT_REFERENCE_PATTERNS)
    is_semantic = _contains_any(lower, SEMANTIC_PATTERNS)

    exact_terms = _unique(match.group(0) for match in EXACT_TOKEN_RE.finditer(text))
    if exact_terms:
        is_exact = True

    if is_summary:
        query_types.append("summary")
    if is_exact:
        query_types.append("exact")
    if is_table:
        query_types.append("table")
    if is_document_reference:
        query_types.append("document_reference")
    if is_semantic or not query_types:
        query_types.append("semantic")

    primary: QueryType
    if is_summary:
        primary = "summary"
    elif is_table:
        primary = "table"
    elif is_exact:
        primary = "exact"
    elif is_document_reference:
        primary = "document_reference"
    elif is_semantic:
        primary = "semantic"
    else:
        primary = "general"

    return QueryAnalysis(
        query=text,
        query_type=primary,
        query_types=query_types,
        exact_terms=exact_terms,
        is_summary=is_summary,
        is_exact=is_exact,
        is_table=is_table,
        is_document_reference=is_document_reference,
    )


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
