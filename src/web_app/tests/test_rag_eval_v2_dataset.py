import json
from collections import Counter
from pathlib import Path

from scripts.build_rag_eval_v2_dataset import (
    DOCUMENTS,
    EXPECTED_CATEGORY_COUNTS,
    build_cases,
    validate_dataset,
)


DATASET_DIR = Path("src/web_app/tests/fixtures/rag_eval_v2")


def test_generated_v2_dataset_is_internally_consistent():
    summary = validate_dataset(build_cases())
    assert summary["documents"] == 22
    assert summary["cases"] == 100
    assert summary["split_counts"] == {"dev": 70, "test": 30}
    assert summary["category_counts"] == EXPECTED_CATEGORY_COUNTS


def test_materialized_v2_dataset_matches_generator():
    rows = []
    for split in ("dev", "test"):
        path = DATASET_DIR / f"cases.{split}.jsonl"
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())

    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    assert Counter(row["split"] for row in rows) == {"dev": 70, "test": 30}
    assert Counter(row["category"] for row in rows) == EXPECTED_CATEGORY_COUNTS
    assert {path.name for path in (DATASET_DIR / "corpus").iterdir()} == set(DOCUMENTS)


def test_gold_facts_exist_in_their_source_documents():
    for split in ("dev", "test"):
        path = DATASET_DIR / f"cases.{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for gold in row["gold_evidence"]:
                content = (DATASET_DIR / "corpus" / gold["filename"]).read_text(encoding="utf-8").lower()
                assert all(str(fact).lower() in content for fact in gold["required_facts"]), row["id"]


def test_unanswerable_cases_have_no_gold_answer_or_evidence():
    cases = build_cases()
    unanswerable = [item for item in cases if item["category"] == "unanswerable"]
    assert len(unanswerable) == 10
    assert all(item["gold_answer"] is None for item in unanswerable)
    assert all(item["gold_evidence"] == [] for item in unanswerable)
