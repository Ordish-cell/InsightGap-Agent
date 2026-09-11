from src.web_app.rag.evidence import assemble_evidence, evidence_block, validate_citations, fit_evidence


def hit(content, parent, **kwargs):
    return {"document_id": "1", "chunk_id": "c1", "parent_id": "p1", "content": content,
            "parent_context": parent, "source_title": "test.md", "metadata": {}, **kwargs}


def test_parent_tail_survives_budget():
    rows = assemble_evidence([hit("ZX-912 价格 9999", "背景" * 4000 + "ZX-912 价格 9999")], query="ZX-912", byte_budget=500)
    assert "9999" in rows[0]["quote"]
    assert len(rows[0]["quote"].encode()) <= 500
    assert rows[0]["context_truncated"]


def test_missing_offsets_use_child_not_unrelated_parent():
    row = assemble_evidence([hit("真正命中", "旧数据的不相关父片段")])[0]
    assert row["quote"] == "真正命中"


def test_multiple_children_keep_both_locations_and_merge_overlap():
    parent = "前" * 2000 + "第一证据" + "中" * 2000 + "第二证据" + "后" * 2000
    row = hit("第一证据", parent, matched_children=[{"chunk_id": "a", "content": "第一证据"}, {"chunk_id": "b", "content": "第二证据"}])
    result = assemble_evidence([row], byte_budget=500)[0]
    assert "第一证据" in result["quote"] and "第二证据" in result["quote"]
    assert result["citation"]["matched_child_ids"] == ["a", "b"]


def test_table_headers_and_precise_source_survive():
    row = hit("north | 12", "背景" * 1000 + "north | 12", metadata={"header": ["region", "quantity"], "sheet_name": "库存", "row_start": 7, "row_end": 7})
    evidence = assemble_evidence([row], byte_budget=400)
    assert "quantity" in evidence[0]["quote"] and "north | 12" in evidence[0]["quote"]
    block = evidence_block(evidence)
    assert "[E1]" in block and '"row_start": 7' in block
    assert validate_citations("结论 [E1] [E2]", ["E1"])["invalid_ids"] == ["E2"]


def test_reassembly_preserves_provenance_and_budget():
    first = assemble_evidence([hit("late fact", "prefix " * 1000 + "late fact", rerank_score=.8, ranking_method="model")], query="late fact", byte_budget=300)
    second = assemble_evidence(first, query="late fact", byte_budget=300)
    assert first[0]["quote"] == second[0]["quote"]
    assert second[0]["ranking_method"] == "model"
    assert second[0]["rerank_score"] == .8


def test_prompt_budget_keeps_top_evidence_instead_of_dropping_everything():
    hits = [hit("needle " * 100, "needle " * 100, document_id=str(i), citation={"matched_terms": ["noise"] * 100}) for i in range(5)]
    selected, block = fit_evidence(hits, "needle", 1500)
    assert selected and selected[0]["document_id"] == "0"
    assert len(block.encode()) <= 1500
    assert "matched_terms" not in block
