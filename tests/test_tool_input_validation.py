"""Test tool input validation."""
import pytest
from src.web_app.mcp.tool_router import validate_tool_input


class TestToolInputValidation:
    def test_email_send_complete_args_valid(self):
        args = {"to": "yu@qq.com", "subject": "高考", "body": "哈哈哈"}
        cleaned, missing = validate_tool_input("email.send", args)
        assert cleaned["to"] == "yu@qq.com"
        assert cleaned["subject"] == "高考"
        assert cleaned["body"] == "哈哈哈"
        assert len(missing) == 0

    def test_email_send_missing_subject_and_body(self):
        args = {"to": "test@example.com"}
        cleaned, missing = validate_tool_input("email.send", args)
        assert cleaned["to"] == "test@example.com"
        missing_fields = {m["field"] for m in missing}
        assert "subject" in missing_fields
        assert "body" in missing_fields
        # Each missing field has a question
        for m in missing:
            assert "field" in m
            assert "question" in m
            assert len(m["question"]) > 0

    def test_email_send_missing_to(self):
        args = {"subject": "Hello", "body": "World"}
        cleaned, missing = validate_tool_input("email.send", args)
        assert cleaned.get("subject") == "Hello"
        assert cleaned.get("body") == "World"
        missing_fields = {m["field"] for m in missing}
        assert "to" in missing_fields

    def test_email_send_invalid_to_format(self):
        args = {"to": "not-an-email", "subject": "Hi", "body": "Test"}
        cleaned, missing = validate_tool_input("email.send", args)
        # 'to' should be rejected due to missing @ with domain
        assert "to" not in cleaned
        missing_fields = {m["field"] for m in missing}
        assert "to" in missing_fields

    def test_email_send_valid_to_with_at(self):
        args = {"to": "user@host.com", "subject": "Hi", "body": "Test"}
        cleaned, missing = validate_tool_input("email.send", args)
        assert cleaned["to"] == "user@host.com"
        assert len(missing) == 0

    def test_local_file_write_complete_args(self):
        args = {"path": "hello.txt", "content": "hello world"}
        cleaned, missing = validate_tool_input("local_file.write", args)
        assert cleaned["path"] == "hello.txt"
        assert cleaned["content"] == "hello world"
        assert len(missing) == 0

    def test_local_file_write_missing_content(self):
        args = {"path": "test.txt"}
        cleaned, missing = validate_tool_input("local_file.write", args)
        assert cleaned["path"] == "test.txt"
        missing_fields = {m["field"] for m in missing}
        assert "content" in missing_fields

    def test_local_file_read_complete(self):
        args = {"path": "hello.txt"}
        cleaned, missing = validate_tool_input("local_file.read", args)
        assert cleaned["path"] == "hello.txt"
        assert len(missing) == 0

    def test_local_file_list_optional_path(self):
        args = {}
        cleaned, missing = validate_tool_input("local_file.list", args)
        # path is optional, so no missing fields
        assert len(missing) == 0

    def test_local_file_append_complete(self):
        args = {"path": "log.txt", "content": "new line"}
        cleaned, missing = validate_tool_input("local_file.append", args)
        assert cleaned["path"] == "log.txt"
        assert cleaned["content"] == "new line"
        assert len(missing) == 0

    def test_unknown_tool_returns_empty_missing(self):
        args = {"foo": "bar"}
        cleaned, missing = validate_tool_input("nonexistent.tool", args)
        assert cleaned == args
        assert missing == []

    def test_optional_fields_preserved(self):
        args = {"to": "a@b.com", "subject": "Hi", "body": "Test", "cc": "cc@b.com"}
        cleaned, missing = validate_tool_input("email.send", args)
        assert cleaned.get("cc") == "cc@b.com"
        assert len(missing) == 0

    def test_empty_string_treated_as_missing(self):
        args = {"to": "  ", "subject": "Hi", "body": "Test"}
        cleaned, missing = validate_tool_input("email.send", args)
        missing_fields = {m["field"] for m in missing}
        assert "to" in missing_fields
