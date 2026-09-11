from types import SimpleNamespace
from src.web_app.rag.index_text import retrieval_text
from scripts.rebuild_rag_context_index import validate_parents
import pytest


def test_index_prefix_keeps_original_text_and_ignores_upload_time():
    chunk = {"content": "原始引用", "metadata": {"heading_path": ["说明"], "version": "v2"}}
    result = retrieval_text(chunk, "doc.md", {"created_at": "2026-09-10"}, mode="metadata")
    assert result.endswith("原始引用") and "v2" in result and "说明" in result
    assert "2026-09-10" not in result
    assert chunk["content"] == "原始引用"


def test_prefix_never_displaces_raw_content_at_embedding_limit():
    text = "x" * 8000
    assert retrieval_text({"content": text}, "large.md", mode="metadata") == text


def test_migration_requires_parent():
    with pytest.raises(ValueError):
        validate_parents([SimpleNamespace(metadata_json={"chunk_role": "child", "parent_id": "missing"})])
