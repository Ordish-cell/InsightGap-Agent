"""Test LLM tool selection integration (infer_tool with llm_result parameter)."""
import pytest

from src.web_app.agent.runtime.intent_schema import LLMToolCall, LLMToolSelectionResult
from src.web_app.mcp.tool_router import _build_email_input, _parse_email_fields, infer_tool


def _make_llm_email_result(to="yu@qq.com", subject="高考", body="哈哈哈", confidence=0.95):
    return LLMToolSelectionResult(
        route="tool",
        confidence=confidence,
        tool_calls=[LLMToolCall(
            name="email.send",
            arguments={"to": to, "subject": subject, "body": body},
        )],
        missing_fields=[],
        requested_action="send_email",
        reason="用户要求发送邮件",
    )


def _make_llm_file_write_result(path="hello.txt", content="hello world", confidence=0.95):
    return LLMToolSelectionResult(
        route="tool",
        confidence=confidence,
        tool_calls=[LLMToolCall(
            name="local_file.write",
            arguments={"path": path, "content": content},
        )],
        missing_fields=[],
        requested_action="write_file",
        reason="用户要求创建文件",
    )


class TestInferToolWithLLMResult:
    """Test that infer_tool correctly uses LLM results."""

    def test_llm_email_result_used(self):
        llm = _make_llm_email_result()
        tool_name, tool_input = infer_tool(
            "帮我给 yu@qq.com 发送个邮件，主题是高考，正文是哈哈哈",
            {},
            llm_result=llm,
        )
        assert tool_name == "email.send"
        assert tool_input["to"] == "yu@qq.com"
        assert tool_input["subject"] == "高考"
        assert tool_input["body"] == "哈哈哈"

    def test_llm_email_alias_normalized(self):
        """LLM returns Chinese alias tool name, should be normalized."""
        llm = LLMToolSelectionResult(
            route="tool", confidence=0.9,
            tool_calls=[LLMToolCall(name="发送邮件", arguments={"to": "x@y.com", "subject": "Hi", "body": "Test"})],
            reason="test",
        )
        tool_name, tool_input = infer_tool("发邮件给 x@y.com", {}, llm_result=llm)
        assert tool_name == "email.send"
        assert tool_input["to"] == "x@y.com"

    def test_llm_low_confidence_falls_back_to_keywords(self):
        """LLM confidence < 0.5 should fall back to keyword matching."""
        llm = _make_llm_email_result(confidence=0.3)
        # Use text that contains a keyword trigger "发邮件"
        tool_name, tool_input = infer_tool(
            "发邮件给 yu@qq.com",
            {},
            llm_result=llm,
        )
        # Should still work via keyword fallback ("发邮件" triggers email.send)
        assert tool_name == "email.send"

    def test_no_llm_result_uses_keywords(self):
        """Without llm_result, infer_tool should use keyword matching as before."""
        tool_name, tool_input = infer_tool("帮我发送邮件给 test@example.com", {})
        assert tool_name == "email.send"
        assert "test@example.com" in tool_input.get("to", "")

    def test_llm_file_write_result_used(self):
        llm = _make_llm_file_write_result()
        tool_name, tool_input = infer_tool(
            "在 workspace 创建 hello.txt，内容是 hello world",
            {},
            llm_result=llm,
        )
        assert tool_name == "local_file.write"
        assert tool_input["path"] == "hello.txt"
        assert tool_input["content"] == "hello world"

    def test_llm_file_read_alias_normalized(self):
        llm = LLMToolSelectionResult(
            route="tool", confidence=0.9,
            tool_calls=[LLMToolCall(name="读取文件", arguments={"path": "test.txt"})],
            reason="test",
        )
        tool_name, tool_input = infer_tool("读取 test.txt", {}, llm_result=llm)
        assert tool_name == "local_file.read"
        assert tool_input["path"] == "test.txt"

    def test_llm_file_list_alias_normalized(self):
        llm = LLMToolSelectionResult(
            route="tool", confidence=0.9,
            tool_calls=[LLMToolCall(name="ls", arguments={"path": "."})],
            reason="test",
        )
        tool_name, tool_input = infer_tool("列出文件", {}, llm_result=llm)
        assert tool_name == "local_file.list"

    def test_llm_result_with_no_tool_calls_uses_keywords(self):
        llm = LLMToolSelectionResult(route="chat", confidence=0.9, tool_calls=[], reason="just chatting")
        tool_name, _ = infer_tool("帮我发送邮件", {}, llm_result=llm)
        # Should use keyword fallback since LLM didn't select a tool
        assert tool_name == "email.send"

    def test_payload_tool_name_overrides_everything(self):
        llm = _make_llm_email_result()
        tool_name, _ = infer_tool(
            "whatever",
            {"tool_name": "local_file.list", "tool_input": {"path": "/tmp"}},
            llm_result=llm,
        )
        assert tool_name == "local_file.list"

    def test_llm_missing_subject_goes_to_missing_fields(self):
        llm = LLMToolSelectionResult(
            route="tool", confidence=0.9,
            tool_calls=[LLMToolCall(name="email.send", arguments={"to": "test@example.com"})],
            missing_fields=[
                {"tool_name": "email.send", "field": "subject", "question": "邮件主题是什么？"},
                {"tool_name": "email.send", "field": "body", "question": "邮件正文是什么？"},
            ],
            reason="missing fields",
        )
        tool_name, tool_input = infer_tool("给 test@example.com 发邮件", {}, llm_result=llm)
        assert tool_name == "email.send"
        assert tool_input["to"] == "test@example.com"
        assert tool_input.get("subject", "") == ""
        assert tool_input.get("body", "") == ""


class TestEmailParser:
    """Test the regex-based email field parser (used as fallback)."""

    def test_full_chinese_email_parse(self):
        text = "帮我给 yu@qq.com 发送个邮件，主题是高考，正文是哈哈哈"
        fields = _parse_email_fields(text)
        assert fields["to"] == "yu@qq.com"
        assert fields["subject"] == "高考"
        assert fields["body"] == "哈哈哈"

    def test_email_with_colons(self):
        text = "发邮件给 eee@example.com，主题: bbbbb，正文: This is a ttt"
        fields = _parse_email_fields(text)
        assert fields["to"] == "eee@example.com"
        assert "bbbbb" in fields["subject"]
        assert "This is a ttt" in fields["body"]

    def test_email_subject_missing(self):
        text = "给我发个邮件，收件人 test@example.com，就说今晚开会"
        fields = _parse_email_fields(text)
        assert fields["to"] == "test@example.com"
        # body should capture the text after 说
        assert "今晚开会" in fields.get("body", "") or len(fields.get("subject", "")) > 0

    def test_bare_email_address_extraction(self):
        text = "发给 yu@qq.com 一封测试邮件"
        fields = _parse_email_fields(text)
        assert fields["to"] == "yu@qq.com"

    def test_email_build_input_includes_user_text_as_body_fallback(self):
        result = _build_email_input("通知 test@example.com 明天开会", {"to": "", "subject": "", "body": ""})
        assert result["to"] == "test@example.com"
        # body falls back to user_input
        assert len(result["body"]) > 0


class TestMissingFieldsStopFlow:
    """Test that missing fields prevent approval/execution and generate questions."""

    def test_email_missing_subject_body_stops(self):
        """LLM result with email.send but no subject/body → missing_fields, no execution."""
        llm = LLMToolSelectionResult(
            route="tool", confidence=0.95,
            tool_calls=[LLMToolCall(name="email.send", arguments={"to": "test@example.com"})],
            missing_fields=[
                {"tool_name": "email.send", "field": "subject", "question": "邮件主题是什么？"},
                {"tool_name": "email.send", "field": "body", "question": "邮件正文是什么？"},
            ],
            reason="用户要发送邮件但缺少参数",
        )
        tool_name, tool_input = infer_tool("给 test@example.com 发封邮件", {}, llm_result=llm)
        # Should identify email.send correctly
        assert tool_name == "email.send"
        assert tool_input["to"] == "test@example.com"
        # validate_tool_input should flag missing subject and body
        from src.web_app.mcp.tool_router import validate_tool_input
        _, missing = validate_tool_input("email.send", tool_input)
        missing_fields = {m["field"] for m in missing}
        assert "subject" in missing_fields
        assert "body" in missing_fields

    def test_email_body_provided_subject_missing(self):
        """LLM extracts body from '发邮件说今晚开会' but subject is missing."""
        llm = LLMToolSelectionResult(
            route="tool", confidence=0.95,
            tool_calls=[LLMToolCall(
                name="email.send",
                arguments={"to": "test@example.com", "body": "今晚开会"},
            )],
            missing_fields=[
                {"tool_name": "email.send", "field": "subject", "question": "邮件主题是什么？"},
            ],
            reason="用户要发邮件说今晚开会，缺少主题",
        )
        tool_name, tool_input = infer_tool("给 test@example.com 发封邮件说今晚开会", {}, llm_result=llm)
        assert tool_name == "email.send"
        assert tool_input["to"] == "test@example.com"
        assert "今晚开会" in str(tool_input.get("body", ""))
        from src.web_app.mcp.tool_router import validate_tool_input
        _, missing = validate_tool_input("email.send", tool_input)
        missing_fields = {m["field"] for m in missing}
        assert "subject" in missing_fields
        # body should NOT be in missing_fields since the LLM extracted it
        assert "body" not in missing_fields


class TestMissingFieldsAnswerBuilder:
    """Test natural language answer generation for missing fields."""

    def test_email_missing_subject_body_answer(self):
        from src.web_app.agent.runtime.nodes import _build_missing_fields_answer
        provided = {"to": "test@example.com"}
        missing = [
            {"field": "subject", "question": "邮件主题是什么？"},
            {"field": "body", "question": "邮件正文是什么？"},
        ]
        answer = _build_missing_fields_answer("email.send", provided, missing)
        assert "test@example.com" in answer
        assert "主题" in answer
        assert "正文" in answer
        # Must NOT contain "已执行" or similar execution language
        assert "已执行" not in answer
        assert "已发送" not in answer

    def test_email_body_provided_subject_missing_answer(self):
        from src.web_app.agent.runtime.nodes import _build_missing_fields_answer
        provided = {"to": "test@example.com", "body": "今晚开会"}
        missing = [
            {"field": "subject", "question": "邮件主题是什么？"},
        ]
        answer = _build_missing_fields_answer("email.send", provided, missing)
        assert "test@example.com" in answer
        assert "今晚开会" in answer
        assert "主题" in answer
        # Must NOT mention body as missing
        assert "正文是什么" not in answer

    def test_file_write_missing_content_answer(self):
        from src.web_app.agent.runtime.nodes import _build_missing_fields_answer
        provided = {"path": "hello.txt"}
        missing = [
            {"field": "content", "question": "文件内容是什么？"},
        ]
        answer = _build_missing_fields_answer("local_file.write", provided, missing)
        assert "hello.txt" in answer
        assert "内容" in answer


class TestToolNotFoundPrevention:
    """Test that obvious email intent never results in tool_not_found."""

    def test_llm_result_with_email_prevents_not_found(self):
        """When LLM selects email.send (even with missing fields), infer_tool returns it."""
        llm = LLMToolSelectionResult(
            route="tool", confidence=0.9,
            tool_calls=[LLMToolCall(name="email.send", arguments={"to": "yu@qq.com"})],
            missing_fields=[
                {"tool_name": "email.send", "field": "subject", "question": "?"},
                {"tool_name": "email.send", "field": "body", "question": "?"},
            ],
            reason="用户要发邮件但缺参数",
        )
        tool_name, _ = infer_tool("帮我给 yu@qq.com 发邮件", {}, llm_result=llm)
        assert tool_name == "email.send"

    def test_keyword_fallback_prevents_not_found_for_email(self):
        """Keyword matching should catch email intent even without LLM."""
        # Use "发送邮件" which is an exact keyword trigger
        tool_name, _ = infer_tool("帮我发送邮件给 yu@qq.com，主题是高考，正文是哈哈哈", {})
        assert tool_name == "email.send"
        assert "yu@qq.com" in str(_)

    def test_chat_text_does_not_trigger_tool(self):
        """Plain chat should not select a tool."""
        tool_name, _ = infer_tool("你好", {})
        assert tool_name is None

    def test_research_text_does_not_trigger_tool_via_keywords(self):
        """Research request should not match tool keywords."""
        tool_name, _ = infer_tool("帮我研究一下 Qwen Agent 最新进展", {})
        assert tool_name is None
