"""Deterministic result checks; this module cannot select or retry actions."""

import re
from pathlib import Path

from src.web_app.rag.evidence import validate_citations


def check_answer(nodes, state, answer):
    evidence = [e for r in state.get("observations", []) for e in r.get("evidence", [])]
    citation = validate_citations(answer, [e.get("evidence_id") for e in evidence])
    notes = []
    if citation["invalid_ids"]:
        notes.append(
            "引用校验未通过："
            + ", ".join(citation["invalid_ids"])
            + " 没有对应证据，请勿将这些引用视为已核实来源。"
        )
    elif (
        evidence
        and citation["status"] == "not_cited"
        and any(r["capability"] == "rag" for r in state.get("observations", []))
    ):
        notes.append("本轮有检索证据，但回答未标明对应引用，事实对应关系尚未核实。")
    if re.search(
        r"已记住|记住了|已记录.*(?:偏好|记忆)|saved.*memory|remembered", answer, re.I
    ):
        receipts = state.get("memory_save_results", [])
        if not any(r.get("ok") for r in receipts):
            notes.append("记忆状态更正：没有确认写入成功，不能视为已经记住。")
        elif not any(r.get("qdrant_indexed") for r in receipts if r.get("ok")):
            notes.append("记忆已保存，但向量索引暂不可用，语义搜索可能受限。")
    if nodes.db is not None:
        from src.web_app.db.repositories.artifact_repository import ArtifactRepository

        for artifact in state.get("artifacts", []):
            row = ArtifactRepository(nodes.db).get_by_user(
                state["user_id"], artifact.get("id")
            )
            if not row or not row.file_path or not Path(row.file_path).is_file():
                notes.append(
                    f"成果 {artifact.get('id')} 未通过存在性检查，不能确认文件可用。"
                )
    for result in state.get("observations", []):
        data = result.get("data", {})
        output = data.get("output", {})
        if output.get("provider") == "mock" and output.get("sent") is False:
            notes.append("邮件执行结果是模拟记录，邮件没有真实发送。")
        if result["status"] in {"unknown", "rejected", "blocked"}:
            notes.append(
                f"动作 {result['capability']} 的状态是 {result['status']}，没有确认执行成功。"
            )
    action = state.get("current_action") or {}
    if action.get("requires_quality_gate") or action.get("confidence", 1) < 0.5:
        if not evidence and not state.get("artifacts") and not state.get("tool_calls"):
            notes.append("本轮缺少可核验的结果，以上回答完整性尚未得到确认。")
    notes = list(dict.fromkeys(notes))
    state.setdefault("final_warnings", []).extend(
        n for n in notes if n not in state.get("final_warnings", [])
    )
    state["evaluation"] = {
        "citation_check": citation,
        "warnings": notes,
        "semantic_support_checked": False,
    }
    return "\n\n" + "\n".join(notes) if notes else ""


def public_result(value):
    if isinstance(value, dict):
        return {
            k: public_result(v)
            for k, v in value.items()
            if not k.startswith("_")
            and not any(
                word in k.lower()
                for word in (
                    "password",
                    "secret",
                    "api_key",
                    "authorization",
                    "reasoning",
                    "chain_of_thought",
                    "raw_prompt",
                )
            )
        }
    if isinstance(value, list):
        return [public_result(v) for v in value]
    return value
