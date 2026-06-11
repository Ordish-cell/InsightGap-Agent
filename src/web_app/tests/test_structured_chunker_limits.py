from src.web_app.rag.structured_chunker import MAX_CHILD_CHARS, build_structured_chunks


def test_structured_chunker_splits_oversized_docx_child():
    long_paragraph = "这是一个很长的段落。" * 1200

    result = build_structured_chunks(long_paragraph, file_type="docx", filename="long.docx")

    vector_chunks = result["vector_chunks"]
    assert len(vector_chunks) > 1
    assert all(len(chunk["content"]) <= MAX_CHILD_CHARS for chunk in vector_chunks)
    assert all((chunk["metadata"] or {}).get("chunk_role") == "child" for chunk in vector_chunks)
    assert result["stats"]["chunk_count"] == len(vector_chunks)


def test_structured_chunker_splits_oversized_table_child():
    header = "name,email,status"
    rows = [f"user-{index},user-{index}@example.com,pending" for index in range(700)]
    csv_text = "\n".join([header, *rows])

    result = build_structured_chunks(csv_text, file_type="csv", filename="large.csv")

    vector_chunks = result["vector_chunks"]
    assert vector_chunks
    assert all(len(chunk["content"]) <= MAX_CHILD_CHARS for chunk in vector_chunks)
    assert any((chunk["metadata"] or {}).get("chunk_type") == "row_block" for chunk in vector_chunks)
