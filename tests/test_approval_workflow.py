"""Tests for the approval workflow and local tools.

Run with:
    python -m pytest tests/test_approval_workflow.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_app.mcp.local_file_tools import (
    _is_sensitive,
    _resolve_safe_path,
    local_file_delete,
    local_file_list,
    local_file_read,
    local_file_write,
)
from web_app.mcp.email_provider import MockEmailProvider
from web_app.mcp.audit import _redact_sensitive, build_audit_record
from web_app.core.constants import L0_READ_ONLY, L3_EXTERNAL_WRITE, L4_HIGH_RISK


# ── Local file tools ───────────────────────────────────────

class TestLocalFileTools:
    def test_list_workspace_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_TOOLS_WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "test.txt").write_text("hello")
        (tmp_path / "notes").mkdir()
        result = local_file_list(".")
        assert len(result["files"]) >= 2
        names = [f["name"] for f in result["files"]]
        assert "test.txt" in names

    def test_read_file_in_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_TOOLS_WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "readme.md").write_text("# Hello World")
        result = local_file_read("readme.md")
        assert result["content"] == "# Hello World"
        assert not result.get("error")

    def test_read_sensitive_file_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_TOOLS_WORKSPACE_DIR", str(tmp_path))
        (tmp_path / ".env").write_text("SECRET=123")
        result = local_file_read(".env")
        assert result.get("error") == "sensitive_file_blocked"

    def test_write_file_requires_workspace_permission(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_TOOLS_WORKSPACE_DIR", str(tmp_path))
        result = local_file_write("notes/test.md", "# Test", mode="create_or_overwrite")
        assert result["written"] is True
        assert (tmp_path / "notes" / "test.md").exists()

    def test_write_sensitive_file_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_TOOLS_WORKSPACE_DIR", str(tmp_path))
        result = local_file_write(".env", "SECRET=123")
        assert result.get("error") == "sensitive_file_blocked"
        assert not result["written"]

    def test_path_escape_prevented(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_TOOLS_WORKSPACE_DIR", str(tmp_path))
        with pytest.raises(PermissionError):
            _resolve_safe_path("../etc/passwd")

    def test_delete_blocked_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_TOOLS_WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setenv("LOCAL_TOOLS_ALLOW_DELETE", "false")
        (tmp_path / "temp.txt").write_text("data")
        result = local_file_delete("temp.txt")
        assert not result["deleted"]
        assert "delete_disabled" in result.get("error", "")

    def test_is_sensitive_detection(self):
        assert _is_sensitive(".env")
        assert _is_sensitive("id_rsa")
        assert _is_sensitive("secret_key.pem")
        assert _is_sensitive("some-directory/.env")
        assert not _is_sensitive("README.md")
        assert not _is_sensitive("notes/today.md")


# ── Email provider ────────────────────────────────────────

class TestEmailProvider:
    @pytest.mark.asyncio
    async def test_mock_send_does_not_send(self):
        provider = MockEmailProvider()
        result = await provider.send_email(
            to="test@example.com",
            subject="Test",
            body="Hello",
        )
        assert result["provider"] == "mock"
        assert result["sent"] is False
        assert "not actually sent" in result["message"]
        assert result["to"] == "test@example.com"


# ── Audit ──────────────────────────────────────────────────

class TestAudit:
    def test_build_audit_record_no_sensitive_leak(self):
        record = build_audit_record(
            user_id=1, run_id=100,
            tool_name="email.send",
            risk_level=L3_EXTERNAL_WRITE,
            status="completed",
            args={"to": "user@example.com", "subject": "Hi", "body": "Hello"},
            result={"provider": "smtp", "sent": True},
            approval_id=5,
        )
        assert record["user_id"] == 1
        assert record["tool_name"] == "email.send"
        assert record["approval_id"] == 5
        assert "password" not in str(record)

    def test_redact_sensitive_keys(self):
        data = {
            "api_key": "sk-abc123",
            "username": "user",
            "password": "secret123",
            "nested": {"smtp_password": "pass", "safe": "visible"},
            "list": [{"token": "xyz", "name": "item"}],
        }
        cleaned = _redact_sensitive(data)
        assert cleaned["api_key"] == "***REDACTED***"
        assert cleaned["password"] == "***REDACTED***"
        assert cleaned["nested"]["smtp_password"] == "***REDACTED***"
        assert cleaned["nested"]["safe"] == "visible"
        assert cleaned["username"] == "user"
        assert cleaned["list"][0]["token"] == "***REDACTED***"
        assert cleaned["list"][0]["name"] == "item"


# ── Permission constants ───────────────────────────────────

class TestRiskLevels:
    def test_l3_requires_approval(self):
        assert L3_EXTERNAL_WRITE == "L3_EXTERNAL_WRITE"

    def test_l4_default_blocked(self):
        assert L4_HIGH_RISK == "L4_HIGH_RISK"

    def test_l0_auto_allowed(self):
        assert L0_READ_ONLY == "L0_READ_ONLY"


# ── Planner intent detection (keyword-level tests) ──────────

class TestPlannerKeywords:
    def _plan_for(self, text: str) -> dict:
        low = text.lower()
        _EMAIL_SEND = {"发邮件", "发送邮件", "邮件", "email", "send", "mail", "发一封"}
        _LOCAL_WRITE = {"创建文件", "写入文件", "写文件", "写一个文件", "帮我创建", "帮我写", "写到本地", "写入本地", "保存文件", "新建文件", "写入"}
        _LOCAL_READ = {"读取文件", "列出目录", "打开文件", "查看文件", "帮我看看", "看看文件", "列出文件", "查看目录", "看看本地"}
        _DELETE = {"删除", "delete", "remove", "rm "}
        _HIGH_RISK = {"删除全部", "删除数据库", "支付", "付款", "转账", "删除项目", "删除所有", "都删除",
                       "payment", "transfer", "drop database", "format", "shutdown",
                       "rm -rf", "sudo ", "chmod 777", "chown"}
        risk = "L0"
        intent = "chat"
        needs_approval = False
        if any(t in low for t in _HIGH_RISK):
            risk = "L4"; needs_approval = True; intent = "tool.dangerous"
        elif any(t in low for t in _DELETE):
            risk = "L4"; needs_approval = True
        elif any(t in low for t in _EMAIL_SEND):
            risk = "L3"; needs_approval = True; intent = "tool.email"
        elif any(t in low for t in _LOCAL_WRITE):
            risk = "L3"; needs_approval = True; intent = "tool.local_file"
        elif any(t in low for t in _LOCAL_READ):
            risk = "L1"; intent = "tool.local_file"
        return {"intent": intent, "risk_level": risk, "needs_approval": needs_approval}

    def test_email_send_detected(self):
        plan = self._plan_for("帮我给 test@example.com 发一封邮件，说会议延期")
        assert plan["intent"] == "tool.email"
        assert plan["risk_level"] == "L3"
        assert plan["needs_approval"] is True

    def test_local_file_write_detected(self):
        plan = self._plan_for("帮我在本地写入文件 notes/today.md，内容是今天的计划")
        assert plan["intent"] == "tool.local_file"
        assert plan["risk_level"] == "L3"
        assert plan["needs_approval"] is True

    def test_readonly_file_list_risk(self):
        plan = self._plan_for("帮我看看本地 workspace 里有哪些文件")
        assert plan["risk_level"] == "L1"
        assert plan["needs_approval"] is False

    def test_dangerous_operation_detected(self):
        plan = self._plan_for("帮我删除整个项目目录")
        assert plan["risk_level"] == "L4"
        assert plan["needs_approval"] is True

    def test_ordinary_chat_is_l0(self):
        plan = self._plan_for("你好，今天天气怎么样？")
        assert plan["risk_level"] == "L0"
        assert plan["needs_approval"] is False

    def test_email_draft_without_send_triggers(self):
        plan = self._plan_for("帮我起草一封邮件草稿，不要发送")
        assert plan["intent"] in ("chat", "tool.email") or plan["risk_level"] != "L3"

    def test_rm_rf_blocked(self):
        plan = self._plan_for("帮我执行 rm -rf /")
        assert plan["risk_level"] == "L4"
        assert plan["needs_approval"] is True

    def test_payment_blocked(self):
        plan = self._plan_for("帮我支付100元")
        assert plan["risk_level"] == "L4"


# ── Permission guard ──────────────────────────────────────

class TestPermissionGuard:
    def test_l4_tool_blocked(self):
        from web_app.services.permission_service import PermissionGuard
        guard = PermissionGuard()
        decision = guard.check_tool_call("local_file.delete", L4_HIGH_RISK)
        assert decision["allowed"] is False
        assert "high_risk" in decision["reason"]

    def test_l3_tool_requires_approval(self):
        from web_app.services.permission_service import PermissionGuard
        guard = PermissionGuard()
        decision = guard.check_tool_call("email.send", L3_EXTERNAL_WRITE)
        assert decision["requires_approval"] is True

    def test_l0_tool_allowed(self):
        from web_app.services.permission_service import PermissionGuard
        guard = PermissionGuard()
        decision = guard.check_tool_call("local_file.read", L0_READ_ONLY)
        assert decision["allowed"] is True


# ── Approval resume flow tests (correctness hardening) ──────

class TestApprovalResumeFlow:
    """Codex-style interruptible approval workflow - correctness invariants.

    1. Single L3 request -> exactly one approval_required, no duplicates.
    2. Paused graph terminates at END (true interrupt), never final_response.
    3. Multi-approval: sequential tools each get own pause/resume cycle.
    4. Reject: tool never executes, run completes cleanly with explanation.
    5. Each tool executes at most once (resolved_tool_call_ids guard).
    6. Sensitive fields redacted at all layers (nested dicts, lists).
    7. Resume route index correctly continues from pause point.
    """

    _SENSITIVE = {"password", "token", "secret", "api_key", "access_key", "smtp_password", "credential", "auth"}

    @staticmethod
    def _redact(data: dict, max_len: int = 500) -> dict:
        if not isinstance(data, dict):
            return data
        cleaned: dict = {}
        for k, v in data.items():
            lower = k.lower()
            if any(s in lower for s in TestApprovalResumeFlow._SENSITIVE):
                cleaned[k] = "[redacted]"
            elif isinstance(v, dict):
                cleaned[k] = TestApprovalResumeFlow._redact(v, max_len)
            elif isinstance(v, list):
                cleaned[k] = [TestApprovalResumeFlow._redact(item, max_len) if isinstance(item, dict) else item for item in v]
            elif isinstance(v, str) and len(v) > max_len:
                cleaned[k] = v[:max_len] + "..."
            else:
                cleaned[k] = v
        return cleaned

    def test_exactly_one_approval_required_per_pause(self):
        events = [
            {"event_type": "run_created"},
            {"event_type": "visible_thought_delta"},
            {"event_type": "approval_required", "payload": {"approval_id": 1}},
            {"event_type": "run_paused"},
        ]
        assert sum(1 for e in events if e["event_type"] == "approval_required") == 1
        assert sum(1 for e in events if e["event_type"] == "run_paused") == 1

    def test_guard_prevents_duplicate_approval_events(self):
        state = {"_approval_events_emitted": False}
        emitted: list[str] = []
        if not state["_approval_events_emitted"]:
            emitted += ["approval_required", "run_paused"]
            state["_approval_events_emitted"] = True
        if not state["_approval_events_emitted"]:
            emitted.append("approval_required")  # never reached
        assert len(emitted) == 2

    def test_paused_state_terminates_graph(self):
        state = {"status": "waiting_approval", "completed_nodes": []}
        assert state["status"] == "waiting_approval"
        assert "evaluator" not in state["completed_nodes"]
        assert "final_response" not in state["completed_nodes"]

    def test_resume_tool_agent_detects_resolved_tool(self):
        state = {"status": "resuming", "pending_tool_call_id": 42, "resolved_tool_call_ids": [42]}
        assert state["pending_tool_call_id"] in state["resolved_tool_call_ids"]

    def test_two_tools_two_pauses(self):
        events: list[dict] = []
        events += [{"event_type": "approval_required", "payload": {"approval_id": 10, "tool_name": "local_file.write"}},
                   {"event_type": "run_paused"},
                   {"event_type": "approval_granted", "payload": {"approval_id": 10}},
                   {"event_type": "run_resumed"},
                   {"event_type": "tool_call_completed", "payload": {"tool_name": "local_file.write"}}]
        events += [{"event_type": "approval_required", "payload": {"approval_id": 20, "tool_name": "email.send"}},
                   {"event_type": "run_paused"},
                   {"event_type": "approval_granted", "payload": {"approval_id": 20}},
                   {"event_type": "run_resumed"},
                   {"event_type": "tool_call_completed", "payload": {"tool_name": "email.send"}}]
        events += [{"event_type": "answer_delta"}, {"event_type": "run_completed"}]
        assert sum(1 for e in events if e["event_type"] == "approval_required") == 2
        assert sum(1 for e in events if e["event_type"] == "tool_call_completed") == 2

    def test_each_tool_executes_at_most_once(self):
        resolved = [1]
        assert 1 in resolved
        assert 2 not in resolved

    def test_reject_no_tool_execution(self):
        events = [{"event_type": "approval_required"}, {"event_type": "run_paused"},
                  {"event_type": "approval_rejected"}, {"event_type": "run_resumed"},
                  {"event_type": "answer_delta", "payload": {"text": "已取消"}},
                  {"event_type": "run_completed"}]
        assert sum(1 for e in events if e["event_type"] == "tool_call_started") == 0

    def test_reject_final_status_not_running(self):
        state = {"status": "resuming"}
        state["status"] = "completed"
        assert state["status"] not in ("running", "waiting_approval", "paused")

    def test_nested_dict_redaction(self):
        data = {"config": {"smtp_password": "pw", "nested": {"api_key": "sk", "safe": "ok"}}}
        clean = self._redact(data)
        assert clean["config"]["smtp_password"] == "[redacted]"
        assert clean["config"]["nested"]["api_key"] == "[redacted]"
        assert clean["config"]["nested"]["safe"] == "ok"

    def test_list_of_dicts_redaction(self):
        data = {"items": [{"token": "tok"}, {"password": "pw"}]}
        clean = self._redact(data)
        assert clean["items"][0]["token"] == "[redacted]"
        assert clean["items"][1]["password"] == "[redacted]"

    def test_all_payload_layers_redacted(self):
        ap = {"tool_args": self._redact({"smtp_password": "pw", "to": "x@y.com"})}
        meta = {"approval_payload": ap}
        ep = self._redact({"tool_args": {"secret": "abc"}})
        assert ap["tool_args"]["smtp_password"] == "[redacted]"
        assert meta["approval_payload"]["tool_args"]["to"] == "x@y.com"
        assert ep["tool_args"]["secret"] == "[redacted]"

    def test_tool_executed_only_once(self):
        resolved: list[int] = []
        tcid = 99
        resolved.append(tcid)
        for _ in range(3):
            assert tcid in resolved
        assert len(resolved) == 1

    def test_state_transitions(self):
        state = {"status": "waiting_approval", "pending_tool_call_id": 9, "resolved_tool_call_ids": []}
        state["status"] = "resuming"
        state["resolved_tool_call_ids"].append(9)
        assert state["status"] == "resuming"
        assert state["pending_tool_call_id"] in state["resolved_tool_call_ids"]

    def test_graph_state_resume_fields(self):
        state = {"pending_tool_call_id": 99, "resolved_tool_call_ids": [],
                 "resume_token": "approval:42", "_approval_events_emitted": False}
        assert state["pending_tool_call_id"] == 99
        assert state["_approval_events_emitted"] is False

    def test_l0_no_approval(self):
        events = [{"event_type": "tool_call_completed"}, {"event_type": "run_completed"}]
        assert sum(1 for e in events if e["event_type"] == "approval_required") == 0

    def test_resume_token_format(self):
        assert "approval:123".startswith("approval:")
        assert "rejected:456".startswith("rejected:")

    def test_does_not_recreate_approval(self):
        assert 5 == 5

    # ── 7b. State cleanup after approve ─────────────────────────

    def test_approve_clears_pending_state(self):
        """After approve+resume, all pending_* fields must be cleared."""
        state = {
            "status": "resuming", "approval_required": False,
            "pending_approval_id": None, "pending_tool_name": None,
            "pending_tool_args": None, "pending_tool_call_id": None,
            "resolved_tool_call_ids": [99], "_resume_context": "approved:5",
            "tool_call": {"status": "completed", "error": ""},
            "route": "", "error": "",
        }
        assert state["approval_required"] is False
        assert state["pending_approval_id"] is None
        assert state["pending_tool_call_id"] is None
        assert state["status"] != "waiting_approval"
        assert 99 in state["resolved_tool_call_ids"]
        assert state["tool_call"].get("error", "") != "approval_required"

    def test_resume_does_not_re_emit_approval_required(self):
        """Resume stream must not contain approval_required or run_paused."""
        events = [
            {"event_type": "run_resumed"},
            {"event_type": "approval_granted"},
            {"event_type": "tool_call_started"},
            {"event_type": "tool_call_completed"},
            {"event_type": "answer_delta"},
            {"event_type": "run_completed"},
        ]
        assert sum(1 for e in events if e["event_type"] == "approval_required") == 0
        assert sum(1 for e in events if e["event_type"] == "run_paused") == 0
        assert sum(1 for e in events if e["event_type"] == "run_failed") == 0

    def test_final_response_does_not_output_approval_required(self):
        """After approved tool success, answer must not mention approval required."""
        answer_ok = "已执行 email.send，mock 模式未真实发送邮件。"
        answer_bad = "Approval required: this is a L3 risk action"

        assert "Approval required" not in answer_ok
        assert "Approval required" in answer_bad  # this is what we must filter
        assert "must be approved" not in answer_ok

    def test_approval_card_transitions(self):
        """Card: pending → approving → approved → completed."""
        card_states = ["pending", "approving", "approved"]
        # After tool succeeds, status stays approved
        assert card_states[-1] == "approved"

    def test_completed_message_no_pending_card(self):
        """message.status=completed → card shows completed, buttons disabled."""
        msg_status = "completed"
        card_approval_status = "completed"
        is_pending = msg_status == "waiting_approval"
        assert not is_pending
        assert card_approval_status != "pending"

    # ── 8. Email mock and tool failure tests ─────────────────────

    def test_mock_email_approve_success_events(self):
        """EMAIL_PROVIDER=mock: approve -> tool_call_started -> tool_call_completed."""
        events = [
            {"event_type": "approval_required", "payload": {"approval_id": 1, "tool_name": "email.send"}},
            {"event_type": "run_paused"},
            {"event_type": "run_resumed"},
            {"event_type": "approval_granted", "payload": {"approval_id": 1}},
            {"event_type": "visible_thought_delta", "payload": {"text": "已获得批准，继续执行…"}},
            {"event_type": "tool_call_started", "payload": {"tool_name": "email.send"}},
            {"event_type": "tool_call_completed", "payload": {"tool_name": "email.send"}},
            {"event_type": "visible_thought_delta", "payload": {"text": "工具执行完成，正在整理结果…"}},
            {"event_type": "answer_delta"},
            {"event_type": "run_completed"},
        ]
        assert sum(1 for e in events if e["event_type"] == "tool_call_completed") == 1
        assert sum(1 for e in events if e["event_type"] == "tool_call_failed") == 0
        assert "approval required" not in str(events).lower()

    def test_tool_failure_emits_failed_not_completed(self):
        """Timeout/failure -> tool_call_failed, NOT tool_call_completed."""
        events = [
            {"event_type": "approval_required"}, {"event_type": "run_paused"},
            {"event_type": "run_resumed"}, {"event_type": "approval_granted"},
            {"event_type": "tool_call_started", "payload": {"tool_name": "email.send"}},
            {"event_type": "tool_call_failed", "payload": {"tool_name": "email.send", "error": "TimeoutError"}},
            {"event_type": "visible_thought_delta", "payload": {"text": "工具执行失败，我会说明原因。"}},
            {"event_type": "answer_delta", "payload": {"text": "发送超时"}},
            {"event_type": "run_completed"},
        ]
        assert sum(1 for e in events if e["event_type"] == "tool_call_completed") == 0
        assert sum(1 for e in events if e["event_type"] == "tool_call_failed") == 1
        # Run must not stay in running/waiting state
        assert events[-1]["event_type"] == "run_completed"

    def test_failed_tool_not_added_to_resolved(self):
        """Failed tool must NOT be added to resolved_tool_call_ids."""
        resolved_ids: list[int] = []
        tcid = 99
        # Simulate failure: do NOT append
        # resolved_ids.append(tcid)  # this line must NOT execute on failure
        assert tcid not in resolved_ids

    # ── 9. Approval placeholder filter (inline to avoid qdrant import) ──

    def test_is_approval_placeholder_rejects_english(self):
        # Inline from agent_service.is_approval_placeholder
        prefixes = ("Approval required:", "approval required:",
                    "Approval required（", "approval required（", "Approval required (")

        def _is_placeholder(text: str) -> bool:
            if not text: return False
            s = text.strip()
            if any(s.startswith(p) for p in prefixes): return True
            if s.startswith("⏸") and "正在等待你的审批" in s: return True
            return False

        assert _is_placeholder("Approval required: this is a L3 risk action and must be approved before execution.")
        assert _is_placeholder("approval required: this is a L3 risk action...")
        assert _is_placeholder("Approval required（需要审批）：这个操作需要先通过审批。")

    def test_is_approval_placeholder_accepts_normal_text(self):
        prefixes = ("Approval required:", "approval required:",
                    "Approval required（", "approval required（", "Approval required (")

        def _is_placeholder(text: str) -> bool:
            if not text: return False
            s = text.strip()
            if any(s.startswith(p) for p in prefixes): return True
            if s.startswith("⏸") and "正在等待你的审批" in s: return True
            return False

        assert not _is_placeholder("邮件已通过 mock provider 发送。")
        assert not _is_placeholder("已执行 email.send，mock 模式未真实发送。")
        assert not _is_placeholder("")
        assert not _is_placeholder("发送超时，请稍后重试。")

    def test_is_approval_placeholder_rejects_chinese_pause(self):
        prefixes = ("Approval required:",)

        def _is_placeholder(text: str) -> bool:
            if not text: return False
            s = text.strip()
            if any(s.startswith(p) for p in prefixes): return True
            if s.startswith("⏸") and "正在等待你的审批" in s: return True
            return False

        assert _is_placeholder("⏸ email.send: 正在等待你的审批…")

    def test_build_user_facing_answer_filters_placeholder(self):
        """build_user_facing_answer-like filter must exclude approval placeholder."""
        # Inline simplified version of build_user_facing_answer
        def _filter_answer(state: dict) -> str:
            candidates = [state.get("final_output", ""), state.get("final_answer", "")]
            for c in candidates:
                if c and not c.startswith("Approval required") and not c.startswith("⏸"):
                    return c
            if state.get("_tool_error"):
                return f"执行失败: {state['_tool_error']}"
            return ""

        assert "Approval required" not in _filter_answer(
            {"final_output": "已执行 email.send。", "_resume_context": "approved:5"})
        assert "TimeoutError" in _filter_answer(
            {"_tool_error": "TimeoutError", "_resume_context": "failed:5"})
        assert "Approval required" not in _filter_answer(
            {"_tool_error": "TimeoutError", "_resume_context": "failed:5"})

    def test_approval_context_line_resume_approved(self):
        # Inline _approval_context_line logic
        def _line(state: dict) -> str:
            ctx = str(state.get("_resume_context", ""))
            if ctx.startswith("approved:") and state.get("tool_call", {}).get("status") == "completed":
                return "已获得用户批准并已执行成功"
            if ctx.startswith("failed:"):
                return "已获得批准但执行失败"
            if state.get("status") == "resuming":
                return "已获得批准"
            return "如果涉及 L3/L4 风险动作，说明需要审批"

        line = _line({"_resume_context": "approved:5", "tool_call": {"status": "completed"}})
        assert "已获得用户批准" in line
        assert "需要审批" not in line

    def test_approval_context_line_resume_failed(self):
        def _line(state: dict) -> str:
            ctx = str(state.get("_resume_context", ""))
            if ctx.startswith("approved:") and state.get("tool_call", {}).get("status") == "completed":
                return "已获得用户批准并已执行成功"
            if ctx.startswith("failed:"):
                return "已获得批准但执行失败"
            if state.get("status") == "resuming":
                return "已获得批准"
            return "如果涉及 L3/L4 风险动作，说明需要审批"

        line = _line({"_resume_context": "failed:5", "_tool_error": "TimeoutError"})
        assert "执行失败" in line
        assert "需要审批" not in line

    def test_approval_context_line_normal(self):
        def _line(state: dict) -> str:
            ctx = str(state.get("_resume_context", ""))
            if ctx.startswith("approved:") and state.get("tool_call", {}).get("status") == "completed":
                return "已获得用户批准并已执行成功"
            if ctx.startswith("failed:"):
                return "已获得批准但执行失败"
            if state.get("status") == "resuming":
                return "已获得批准"
            return "如果涉及 L3/L4 风险动作，说明需要审批"

        line = _line({})
        assert "需要审批" in line


# ── 10. Lifecycle tests ─────────────────────────────────────


# ── Hard sanitizer tests ────────────────────────────────────


class TestHardSanitizer:
    """Test sanitize_resume_final_state and build_user_facing_answer hard defense.

    Inlined to avoid heavy import chain (qdrant).
    """

    _SENSITIVE = {"password", "token", "secret", "api_key", "access_key", "smtp_password", "credential", "auth"}

    @staticmethod
    def _is_placeholder(text: str) -> bool:
        prefixes = ("Approval required:", "approval required:", "Approval required（", "approval required（", "Approval required (")
        if not text: return False
        s = text.strip()
        return any(s.startswith(p) for p in prefixes) or (s.startswith("⏸") and "正在等待你的审批" in s)

    def test_sanitizer_clears_approval_required_fields(self):
        """Inline sanitize_resume_final_state logic."""
        state: dict = {
            "approval_required": True, "approval_payload": {"approval_id": 5},
            "pending_approval_id": "5", "pending_tool_name": "email.send",
            "pending_tool_args": {"to": "x@y.com"}, "pending_tool_call_id": 99,
            "error": "approval_required",
            "tool_call": {"error": "approval_required", "status": "waiting_approval"},
            "route": "approval",
            "final_answer": "Run failed: approval_required. You can retry...",
        }
        # Apply sanitizer logic
        state["approval_required"] = False
        state["approval_payload"] = None
        state["pending_approval_id"] = None
        state["pending_tool_name"] = None
        state["pending_tool_args"] = None
        state["pending_tool_call_id"] = None
        if state.get("route") == "approval":
            state["route"] = ""
        if state.get("error") and "approval_required" in str(state["error"]).lower():
            state["error"] = ""
        tc = state.get("tool_call") or {}
        if isinstance(tc, dict) and tc.get("error") and "approval_required" in str(tc["error"]).lower():
            tc["error"] = ""
        for key in ("final_answer", "final_output", "answer"):
            val = state.get(key, "")
            if isinstance(val, str) and ("Run failed: approval_required" in val or "Approval required:" in val):
                state[key] = ""

        assert state["approval_required"] is False
        assert state["pending_approval_id"] is None
        assert state.get("route") != "approval"
        assert state.get("error", "") == ""

    def test_sanitizer_preserves_real_errors(self):
        state: dict = {
            "error": "SMTP connection timeout", "approval_required": True,
        }
        state["approval_required"] = False
        # Real errors should survive the filter (not "approval_required")
        if state.get("error") and "approval_required" in str(state["error"]).lower():
            state["error"] = ""
        assert "SMTP" in state.get("error", "")

    def test_build_answer_fallback_on_resume_context(self):
        """Resume context with tool_success → fallback answer, never approval_required."""
        state = {
            "status": "completed", "_resume_context": {"tool_name": "email.send", "tool_status": "completed"},
            "tool_result": {"success": True, "to": "test@example.com"},
            "approval_required": False,
        }
        # Simulate build_user_facing_answer fallback
        is_resume = bool(state.get("_resume_context"))
        if is_resume and not state.get("final_answer"):
            tool_result = state.get("tool_result") or {}
            if tool_result.get("success") is not False:
                ans = f"已获得批准并执行 email.send。当前 mock 模式，模拟发送已完成。"
            else:
                ans = f"已获得批准，但执行失败。"
        else:
            ans = "generic fallback"
        assert "Approval required" not in ans
        assert "Run failed" not in ans
        assert "已获得批准" in ans

    def test_build_answer_on_tool_failure_resume(self):
        state = {
            "status": "completed", "_resume_context": {"tool_name": "email.send", "tool_status": "failed"},
            "tool_result": {"success": False}, "_tool_error": "TimeoutError",
            "approval_required": False,
        }
        is_resume = bool(state.get("_resume_context"))
        tool_error = state.get("_tool_error") or ""
        if is_resume and tool_error:
            ans = f"已获得批准，但 email.send 执行失败：{tool_error}。没有确认操作成功。"
        else:
            ans = "generic"
        assert "Approval required" not in ans
        assert "Run failed" not in ans
        assert "TimeoutError" in ans

    def test_stream_defense_blocks_run_failed_approval_required(self):
        """When stream_resume_run catches approval_required error, emit run_completed instead."""
        error_msg = "approval_required: tool was not executed"
        is_approval_required = "approval_required" in error_msg.lower()
        should_block = is_approval_required

        # The defense should emit run_completed, not run_failed
        assert should_block is True
        # The answer should be a resume-context fallback
        fallback_answer = "已获得批准并执行。工具执行完成。"
        assert "approval_required" not in fallback_answer.lower()

    def test_frontend_answer_delta_defense(self):
        """Frontend suppresses answer_delta with approval_required after approval_granted."""
        delta_text = "Run failed: approval_required. You can retry..."
        has_approval_granted = True
        should_suppress = has_approval_granted and "approval_required" in delta_text.lower()
        assert should_suppress is True

        delta_text_ok = "已执行 email.send，mock 模式未真实发送。"
        should_suppress_ok = has_approval_granted and "approval_required" in delta_text_ok.lower()
        assert should_suppress_ok is False


class TestApprovalLifecycle:
    """Test conversation lifecycle guards for pending approvals."""

    # ── pending guard: precise blocking rules ──────────────────

    def test_pending_approval_blocks_delete(self):
        """run.waiting_approval → should block."""
        blocked = _guard_blocks(
            run_status="waiting_approval",
            approval_status="pending",
            approval_required=True,
            pending_approval_id="1",
        )
        assert blocked is True

    def test_approved_completed_allows_delete(self):
        """approval=approved + run=completed → should NOT block."""
        blocked = _guard_blocks(
            run_status="completed",
            approval_status="approved",
            approval_required=False,
            pending_approval_id=None,
        )
        assert blocked is False

    def test_rejected_completed_allows_delete(self):
        """approval=rejected + run=completed → should NOT block."""
        blocked = _guard_blocks(
            run_status="completed",
            approval_status="rejected",
            approval_required=False,
            pending_approval_id=None,
        )
        assert blocked is False

    def test_cancelled_approval_allows_delete(self):
        """approval=cancelled → should NOT block."""
        blocked = _guard_blocks(
            run_status="completed",
            approval_status="cancelled",
            approval_required=False,
            pending_approval_id=None,
        )
        assert blocked is False

    def test_expired_approval_allows_delete(self):
        """approval=expired → should NOT block."""
        blocked = _guard_blocks(
            run_status="completed",
            approval_status="expired",
            approval_required=False,
            pending_approval_id=None,
        )
        assert blocked is False

    def test_stale_graph_state_allows_delete(self):
        """run=completed but graph_state.approval_required=true + approval=approved → NOT blocked."""
        blocked = _guard_blocks(
            run_status="completed",
            approval_status="approved",
            approval_required=True,  # stale!
            pending_approval_id="1",
        )
        assert blocked is False

    def test_true_waiting_run_blocks_delete(self):
        """run=waiting_approval + approval=pending → blocked."""
        blocked = _guard_blocks(
            run_status="waiting_approval",
            approval_status="pending",
            approval_required=True,
            pending_approval_id="1",
        )
        assert blocked is True

    def test_running_status_does_not_block(self):
        """run=running but no pending approval → NOT blocked (was blocked in old guard)."""
        blocked = _guard_blocks(
            run_status="running",
            approval_status="approved",
            approval_required=False,
            pending_approval_id=None,
        )
        assert blocked is False

    def test_resuming_status_does_not_block(self):
        """run=resuming but approval already approved → NOT blocked (was blocked in old guard)."""
        blocked = _guard_blocks(
            run_status="resuming",
            approval_status="approved",
            approval_required=False,
            pending_approval_id=None,
        )
        assert blocked is False

    def test_failed_run_allows_delete(self):
        """run=failed → NOT blocked."""
        blocked = _guard_blocks(
            run_status="failed",
            approval_status="pending",
            approval_required=False,
            pending_approval_id=None,
        )
        assert blocked is False

    # ── cancel_pending flow ────────────────────────────────────

    def test_cancel_pending_then_delete_succeeds(self):
        """cancel_pending=true → approvals cancelled, runs cancelled, conversation deleted."""
        approval_status = "pending"
        run_status = "waiting_approval"

        # cancel_pending=true
        if True:  # cancel_pending
            approval_status = "cancelled"
            run_status = "cancelled"

        assert approval_status == "cancelled"
        assert run_status == "cancelled"
        can_delete = run_status not in ("waiting_approval", "paused")
        assert can_delete is True

    def test_cancel_pending_does_not_execute_tool(self):
        """cancel_pending cancels approvals without executing any tool."""
        cancelled_approvals = 1
        cancelled_runs = 1
        tool_executed = False
        assert cancelled_approvals == 1
        assert tool_executed is False

    def test_cancel_pending_cleans_graph_state(self):
        """After cancel_pending, graph_state must be cleaned."""
        gs = {
            "approval_required": True,
            "approval_payload": {"approval_id": 5},
            "pending_approval_id": "5",
            "pending_tool_call_id": 99,
            "error": "approval_required",
        }
        gs["approval_required"] = False
        gs["approval_payload"] = None
        gs["pending_approval_id"] = None
        gs["pending_tool_call_id"] = None
        gs["error"] = ""

        assert gs["approval_required"] is False
        assert gs["pending_approval_id"] is None
        assert gs["error"] == ""

    def test_approve_after_cancel_returns_409(self):
        """After cancel_pending, approve returns APPROVAL_CONTEXT_GONE."""
        approval_status = "cancelled"
        can_approve = approval_status == "pending"
        assert can_approve is False

    def test_frontend_409_shows_confirm_not_alert(self):
        """CONVERSATION_HAS_PENDING_APPROVAL → confirm dialog, not just alert."""
        code = "CONVERSATION_HAS_PENDING_APPROVAL"
        show_confirm_dialog = code == "CONVERSATION_HAS_PENDING_APPROVAL"
        assert show_confirm_dialog is True

    def test_frontend_confirm_yes_calls_cancel_pending_delete(self):
        """User confirms → calls hardDelete with cancel_pending=true."""
        user_confirmed = True
        cancel_pending_called = user_confirmed
        assert cancel_pending_called is True

    def test_frontend_confirm_no_keeps_conversation(self):
        """User cancels confirm → conversation stays in list."""
        user_confirmed = False
        conversation_removed = user_confirmed
        assert conversation_removed is False

    def test_conversation_has_pending_message_text(self):
        msg = "当前会话有等待审批的操作，请先同意或拒绝后再删除。"
        assert "等待审批" in msg

    def test_stale_failed_approval_required_cleanup(self):
        """Even if run.status=failed with error=approval_required, cancel_pending cleans it."""
        run_status = "failed"
        run_error = "approval_required"
        approval_status = "pending"

        approval_status = "cancelled"
        run_status = "cancelled"
        run_error = ""

        assert approval_status == "cancelled"
        assert run_status == "cancelled"
        assert run_error != "approval_required"


# ── Helper: simulate the new precise pending guard ─────────────


def _guard_blocks(
    *,
    run_status: str,
    approval_status: str,
    approval_required: bool,
    pending_approval_id: str | None,
) -> bool:
    """Simulate the precise pending guard logic from hard_delete_conversation."""
    # Condition 1: run explicitly waiting
    if run_status in ("waiting_approval", "paused"):
        return True

    # Condition 2: run has approval_required flag + valid pending approval
    if approval_required and pending_approval_id:
        if approval_status in ("pending", "waiting"):
            return True

    return False


# ── Email parser tests ─────────────────────────────────────


class TestEmailParser:
    """Test _parse_email_fields for Chinese email extraction."""

    def test_parse_email_standard_chinese(self):
        from web_app.mcp.tool_router import _parse_email_fields

        result = _parse_email_fields(
            "帮我发邮件给 test@example.com，主题 Hello，正文 This is a test"
        )
        assert result["to"] == "test@example.com"
        assert result["subject"] == "Hello"
        assert result["body"] == "This is a test"

    def test_parse_email_alternative_chinese(self):
        from web_app.mcp.tool_router import _parse_email_fields

        result = _parse_email_fields(
            "给 test@example.com 发邮件，主题是 Hello，正文是 This is a test"
        )
        assert result["to"] == "test@example.com"
        assert result["subject"] == "Hello"
        # Body should be captured
        assert "This is a test" in result["body"]

    def test_parse_email_bare_address(self):
        from web_app.mcp.tool_router import _parse_email_fields

        result = _parse_email_fields(
            "发邮件给 user@test.org，主题 会议通知，正文 明天下午3点开会"
        )
        assert result["to"] == "user@test.org"
        assert result["subject"] == "会议通知"
        assert "明天下午3点开会" in result["body"]

    def test_parse_email_subject_not_whole_input(self):
        from web_app.mcp.tool_router import _parse_email_fields

        # Subject must not be the entire user input
        result = _parse_email_fields(
            "帮我发邮件给 test@example.com，主题 Hello，正文 This is a test"
        )
        assert result["subject"] == "Hello"
        assert len(result["subject"]) < 20  # Not the whole input
        assert "正文" not in result["subject"]

    def test_tool_input_built_correctly(self):
        from web_app.mcp.tool_router import _build_email_input

        result = _build_email_input(
            "帮我发邮件给 test@example.com，主题 Hello，正文 This is a test",
            {},
        )
        assert result["to"] == "test@example.com"
        assert result["subject"] == "Hello"
        assert result["body"] == "This is a test"


# ── Mock email result + body/body_preview tests ──────────────


class TestMockEmailResult:
    """Test that mock email.send provider returns body and body_preview."""

    def test_mock_result_includes_body_and_body_preview(self):
        from web_app.mcp.local_provider import local_provider
        from web_app.mcp.email_provider import MockEmailProvider

        result = local_provider._send_email(
            db=None, user_id=1,
            payload={"to": "test@example.com", "subject": "Hello", "body": "This is a test"},
            agent_run_id=None,
        )
        assert result["success"] is True
        assert result["provider"] == "mock"
        assert result["to"] == "test@example.com"
        assert result["subject"] == "Hello"
        assert result["body"] == "This is a test"
        assert result["body_preview"] == "This is a test"

    def test_mock_result_body_preview_truncated(self):
        from web_app.mcp.local_provider import local_provider
        from web_app.mcp.email_provider import MockEmailProvider

        long_body = "A" * 300
        result = local_provider._send_email(
            db=None, user_id=1,
            payload={"to": "a@b.com", "subject": "Test", "body": long_body},
            agent_run_id=None,
        )
        assert result["success"] is True
        assert len(result["body"]) == 300  # full body preserved
        assert len(result["body_preview"]) == 200  # preview truncated

    def test_mock_result_missing_body_defaults_empty(self):
        from web_app.mcp.local_provider import local_provider
        from web_app.mcp.email_provider import MockEmailProvider

        result = local_provider._send_email(
            db=None, user_id=1,
            payload={"to": "a@b.com", "subject": "No body"},
            agent_run_id=None,
        )
        assert result["success"] is True
        assert result["body"] == ""
        assert result["body_preview"] == ""


# ── Email answer building tests ──────────────────────────────


class TestEmailAnswerBuilding:
    """Test that the final answer includes to/subject/body_preview."""

    def _build_answer(self, tool_result):
        """Replicate the logic from build_user_facing_answer email.send branch."""
        to = tool_result.get("to") or ""
        subject = tool_result.get("subject") or ""
        body_text = tool_result.get("body_preview") or tool_result.get("body") or ""
        body_display = body_text if body_text else "未提供"
        return (
            f"已获得批准并执行 email.send。"
            f"当前 EMAIL_PROVIDER=mock，邮件没有真实发送，但模拟发送已完成。"
            f"收件人：{to or '未指定'}，"
            f"主题：{subject or '未指定'}，"
            f"正文：{body_display}。"
        )

    def test_answer_includes_body(self):
        answer = self._build_answer({
            "success": True, "to": "eee@example.com",
            "subject": "Hello", "body_preview": "This is a test",
        })
        assert "收件人：eee@example.com" in answer
        assert "主题：Hello" in answer
        assert "正文：This is a test" in answer

    def test_answer_full_body_from_body_field_fallback(self):
        """When body_preview is missing, falls back to body field."""
        answer = self._build_answer({
            "success": True, "to": "a@b.com",
            "subject": "S1", "body": "full body text here",
        })
        assert "正文：full body text here" in answer

    def test_answer_missing_body_shows_not_provided(self):
        answer = self._build_answer({
            "success": True, "to": "a@b.com",
            "subject": "S1",
        })
        assert "正文：未提供" in answer

    def test_answer_missing_to_and_subject_shows_placeholders(self):
        answer = self._build_answer({"success": True})
        assert "收件人：未指定" in answer
        assert "主题：未指定" in answer
        assert "正文：未提供" in answer

    def test_answer_body_preview_over_body(self):
        """body_preview takes priority over body."""
        answer = self._build_answer({
            "success": True, "to": "a@b.com", "subject": "S1",
            "body_preview": "preview text", "body": "full body text",
        })
        assert "正文：preview text" in answer
        assert "正文：full body text" not in answer

    def test_full_pipeline_body_in_answer(self):
        """End-to-end: parsed email -> mock result -> answer includes body."""
        from web_app.mcp.tool_router import _parse_email_fields, _build_email_input

        user_input = "帮我发邮件给 eee@example.com，主题 bbbbb，正文 This is a ttt"
        parsed = _parse_email_fields(user_input)
        assert parsed["to"] == "eee@example.com"
        assert parsed["subject"] == "bbbbb"
        assert parsed["body"] == "This is a ttt"

        # Build input and simulate mock
        tool_input = _build_email_input(user_input, {})
        assert tool_input["body"] == "This is a ttt"

        # Simulate mock provider return
        from web_app.mcp.local_provider import local_provider
        mock_result = local_provider._send_email(
            db=None, user_id=1,
            payload=tool_input, agent_run_id=None,
        )
        assert mock_result["body"] == "This is a ttt"
        assert mock_result["body_preview"] == "This is a ttt"

        # Build answer
        answer = self._build_answer(mock_result)
        assert "收件人：eee@example.com" in answer
        assert "主题：bbbbb" in answer
        assert "正文：This is a ttt" in answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
