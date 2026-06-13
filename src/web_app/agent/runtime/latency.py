from __future__ import annotations

from typing import Any

PREFETCH_SLOW_MS = 3000
PARALLEL_READ_SLOW_MS = 6000
RAG_PREPARE_SLOW_MS = 3000
CONTEXT_SKILL_SLOW_MS = 6000


def build_runtime_latency_trace(state: dict[str, Any], elapsed_ms: int | None = None) -> dict[str, Any]:
    """Build a read-only latency and hit-rate snapshot from runtime state."""
    warnings: list[str] = []
    prefetch_results = state.get("prefetch_results") if isinstance(state.get("prefetch_results"), dict) else {}
    parallel_read_results = state.get("parallel_read_results") if isinstance(state.get("parallel_read_results"), dict) else {}
    branch_timings = state.get("parallel_read_branch_timings") if isinstance(state.get("parallel_read_branch_timings"), dict) else {}
    rag_prepare = parallel_read_results.get("rag_prepare") if isinstance(parallel_read_results.get("rag_prepare"), dict) else {}
    dispatch_audit = state.get("supervisor_dispatch_audit") if isinstance(state.get("supervisor_dispatch_audit"), dict) else {}
    readiness = state.get("supervisor_readiness_report") if isinstance(state.get("supervisor_readiness_report"), dict) else {}
    control = state.get("supervisor_control_decision") if isinstance(state.get("supervisor_control_decision"), dict) else {}

    prefetch_warnings = _string_list(state.get("prefetch_warnings"))
    parallel_warnings = _string_list(state.get("parallel_read_warnings"))
    supervisor_warnings = (
        _string_list(state.get("supervisor_dispatch_warnings"))
        + _string_list(state.get("supervisor_readiness_warnings"))
        + _string_list(state.get("supervisor_control_warnings"))
    )
    if prefetch_warnings:
        warnings.append("prefetch_warnings_present")
    if parallel_warnings:
        warnings.append("parallel_read_warnings_present")
    if supervisor_warnings:
        warnings.append("supervisor_warnings_present")

    rag_prepare_evidence = list(rag_prepare.get("evidence") or []) if isinstance(rag_prepare.get("evidence"), list) else []
    evidence_count = _int_value(rag_prepare.get("evidence_count"), len(rag_prepare_evidence))
    rag_prepare_status = str(rag_prepare.get("status") or "")
    search_attempted = bool(rag_prepare.get("search_attempted"))
    rag_prepare_no_evidence = rag_prepare_status == "ok" and search_attempted and evidence_count == 0

    trace = {
        "mode": "runtime_latency_trace",
        "total_elapsed_ms": _optional_int(elapsed_ms),
        "prefetch": {
            "elapsed_ms": _int_value(state.get("prefetch_elapsed_ms"), 0),
            "has_results": bool(prefetch_results),
            "warnings": prefetch_warnings,
            "rag_evidence_count": _count_nested(prefetch_results, "rag", "evidence"),
            "memory_count": _count_nested(prefetch_results, "memory", "items"),
            "skill_matched": bool((prefetch_results.get("skill") or {}).get("matched_skill")) if isinstance(prefetch_results.get("skill"), dict) else False,
        },
        "parallel_read": {
            "elapsed_ms": _int_value(state.get("parallel_read_elapsed_ms"), 0),
            "branch_timings": {str(key): _int_value(value, 0) for key, value in branch_timings.items()},
            "warnings": parallel_warnings,
            "rag_prepare": {
                "status": rag_prepare_status,
                "elapsed_ms": _int_value(rag_prepare.get("elapsed_ms"), 0),
                "from_prefetch": bool(rag_prepare.get("from_prefetch")),
                "search_attempted": search_attempted,
                "evidence_count": evidence_count,
                "no_evidence": rag_prepare_no_evidence,
            },
        },
        "supervisor": {
            "dispatch_status": dispatch_audit.get("status"),
            "legacy_next_node": dispatch_audit.get("legacy_next_node"),
            "readiness_level": readiness.get("readiness_level"),
            "ready_for_control": bool(readiness.get("ready_for_control")),
            "control_applied": bool(control.get("control_applied")),
            "selected_next_node": control.get("selected_next_node"),
            "fallback_reason": control.get("fallback_reason") or "",
        },
        "agent_results": {
            "count": len(state.get("agent_results") or []) if isinstance(state.get("agent_results"), list) else 0,
            "failed_agents": _failed_agents(state.get("agent_results")),
        },
    }
    slow_path_hints = build_runtime_slow_path_hints(trace)
    return {
        "runtime_latency_trace": trace,
        "runtime_latency_warnings": warnings,
        "runtime_slow_path_hints": slow_path_hints,
    }


def build_runtime_slow_path_hints(trace: dict[str, Any]) -> list[str]:
    """Return stable latency diagnosis hints from a runtime latency trace."""
    hints: list[str] = []
    prefetch = trace.get("prefetch") if isinstance(trace.get("prefetch"), dict) else {}
    parallel_read = trace.get("parallel_read") if isinstance(trace.get("parallel_read"), dict) else {}
    rag_prepare = parallel_read.get("rag_prepare") if isinstance(parallel_read.get("rag_prepare"), dict) else {}
    branch_timings = parallel_read.get("branch_timings") if isinstance(parallel_read.get("branch_timings"), dict) else {}
    supervisor = trace.get("supervisor") if isinstance(trace.get("supervisor"), dict) else {}

    if _int_value(prefetch.get("elapsed_ms"), 0) > PREFETCH_SLOW_MS:
        _append_once(hints, "prefetch_slow")
    if prefetch.get("warnings"):
        _append_once(hints, "prefetch_timeout_or_warning")
    if _int_value(parallel_read.get("elapsed_ms"), 0) > PARALLEL_READ_SLOW_MS:
        _append_once(hints, "parallel_read_slow")
    if parallel_read.get("warnings"):
        _append_once(hints, "parallel_read_timeout_or_warning")
    if _int_value(rag_prepare.get("elapsed_ms"), 0) > RAG_PREPARE_SLOW_MS:
        _append_once(hints, "rag_prepare_slow")
    if bool(rag_prepare.get("no_evidence")):
        _append_once(hints, "rag_prepare_no_evidence")
    if _int_value(branch_timings.get("context_skill"), 0) > CONTEXT_SKILL_SLOW_MS:
        _append_once(hints, "context_skill_slow")
    if supervisor.get("fallback_reason"):
        _append_once(hints, "supervisor_control_fallback")
    return hints


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _int_value(value, 0)


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _count_nested(parent: dict[str, Any], key: str, child_key: str) -> int:
    child = parent.get(key) if isinstance(parent, dict) else {}
    if not isinstance(child, dict):
        return 0
    value = child.get(child_key)
    return len(value) if isinstance(value, list) else 0


def _failed_agents(agent_results: Any) -> list[str]:
    if not isinstance(agent_results, list):
        return []
    failed: list[str] = []
    for item in agent_results:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "") in {"failed", "timeout", "denied"}:
            agent = str(item.get("agent") or "")
            if agent and agent not in failed:
                failed.append(agent)
    return failed


def _append_once(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
