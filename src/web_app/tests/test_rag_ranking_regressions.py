from src.web_app.rag.retriever import ParentChildRetriever
from src.web_app.rag.query_analyzer import analyze_query
from src.web_app.tests.test_rag_hybrid_retrieval import hybrid_env, _add_document_with_child, _add_child, _vector_point


def test_rrf_order_survives_keyword_rules_and_refills_parents(hybrid_env):
    db, user, _ = hybrid_env
    first, child = _add_document_with_child(db, user.id, "first.md", "semantic answer", chunk_id="c-1")
    second, child2 = _add_document_with_child(db, user.id, "keyword.md", "keyword keyword", chunk_id="c-1")
    _add_child(db, user.id, first.id, "more semantic", chunk_id="c-2", parent_id="p-1")
    hits = [
        _vector_point(user.id, first.id, "c-1", child.content, score=.8, parent_id="p-1", filename=first.filename),
        _vector_point(user.id, first.id, "c-2", "more semantic", score=.7, parent_id="p-1", filename=first.filename),
        _vector_point(user.id, second.id, "c-1", child2.content, score=.6, parent_id="p-1", filename=second.filename),
    ]
    class Store:
        def search_hybrid(self, **kwargs):
            assert kwargs["top_k"] == 20
            return hits
    trace = {}
    results = ParentChildRetriever(db, Store()).search(user_id=user.id, query="keyword", query_vector=[.1],
        backend="qdrant_hybrid", top_k=2, trace=trace)
    assert [r["document_id"] for r in results] == [str(first.id), str(second.id)]
    assert len(results[0]["matched_children"]) == 2
    assert results[0]["ranking_method"] == "rrf"
    assert results[0]["final_score"] == .8
    assert len(trace["candidates"]) == 3


def test_merge_key_includes_document_id():
    r = ParentChildRetriever(None)
    hits = r._merge_hits([{"document_id": 1, "chunk_id": "c1", "vector_score": .9},
                          {"document_id": 2, "chunk_id": "c1", "vector_score": .8}], [], analyze_query("hello"))
    assert len(hits) == 2


def test_empty_document_scope_never_searches(hybrid_env):
    db, user, _ = hybrid_env
    assert ParentChildRetriever(db).search(user_id=user.id, query="x", document_ids=[]) == []
