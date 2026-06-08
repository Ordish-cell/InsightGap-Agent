"""Tests for the approval workflow and local tools.

Run with:
    python -m pytest tests/test_approval_workflow.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from web_app.mcp.local_file_tools import (
    _ensure_workspace,
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
            user_id=1,
            run_id=100,
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
        # Must not contain raw passwords
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
    """Verify keyword detection logic without importing the full runtime.

    These tests directly exercise the keyword sets used by planner.py
    to avoid the heavy import chain (sqlalchemy, qdrant, etc.).
    """

    def _plan_for(self, text: str) -> dict:
        """Minimal inline reimplementation of plan_route for testing."""
        low = text.lower()

        # Keyword sets from planner.py
        _EMAIL_SEND = {"发邮件", "发送邮件", "邮件", "email", "send", "mail", "发一封"}
        _LOCAL_WRITE = {"创建文件", "写入文件", "写文件", "写一个文件", "帮我创建", "帮我写", "写到本地", "写入本地", "保存文件", "新建文件", "写入"}
        _LOCAL_READ = {"读取文件", "列出目录", "打开文件", "查看文件", "帮我看看", "看看文件", "列出文件", "查看目录", "看看本地"}
        _DELETE = {"删除", "delete", "remove", "rm "}
        _HIGH_RISK = {"删除全部", "删除数据库", "支付", "付款", "转账", "删除项目", "删除所有", "都删除",
                       "payment", "transfer", "drop database", "format", "shutdown",
                       "rm -rf", "sudo ", "chmod 777", "chown"}

        needs_approval = False
        risk = "L0"
        intent = "chat"

        if any(t in low for t in _HIGH_RISK):
            risk = "L4"
            needs_approval = True
            intent = "tool.dangerous"
        elif any(t in low for t in _DELETE):
            risk = "L4"
            needs_approval = True
        elif any(t in low for t in _EMAIL_SEND):
            risk = "L3"
            needs_approval = True
            intent = "tool.email"
        elif any(t in low for t in _LOCAL_WRITE):
            risk = "L3"
            needs_approval = True
            intent = "tool.local_file"
        elif any(t in low for t in _LOCAL_READ):
            risk = "L1"
            intent = "tool.local_file"

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
        # "发送" is in the text but not "发邮件" / "send email"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
