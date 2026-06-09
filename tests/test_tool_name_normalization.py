"""Test tool name alias normalization."""
import pytest
from src.web_app.mcp.registry import normalize_tool_name


class TestToolNameNormalization:
    def test_email_send_chinese_aliases(self):
        assert normalize_tool_name("发送邮件") == "email.send"
        assert normalize_tool_name("发邮件") == "email.send"
        assert normalize_tool_name("寄邮件") == "email.send"
        assert normalize_tool_name("发一封") == "email.send"

    def test_email_send_english_aliases(self):
        assert normalize_tool_name("send_email") == "email.send"
        assert normalize_tool_name("email_send") == "email.send"
        assert normalize_tool_name("mail.send") == "email.send"
        assert normalize_tool_name("send.mail") == "email.send"
        assert normalize_tool_name("sendEmail") == "email.send"
        assert normalize_tool_name("send_mail") == "email.send"

    def test_email_send_case_insensitivity(self):
        assert normalize_tool_name("Send_Email") == "email.send"
        assert normalize_tool_name("SEND.MAIL") == "email.send"
        assert normalize_tool_name("  SendEmail  ") == "email.send"

    def test_local_file_write_aliases(self):
        assert normalize_tool_name("写文件") == "local_file.write"
        assert normalize_tool_name("创建文件") == "local_file.write"
        assert normalize_tool_name("保存文件") == "local_file.write"
        assert normalize_tool_name("write_file") == "local_file.write"
        assert normalize_tool_name("create_file") == "local_file.write"

    def test_local_file_read_aliases(self):
        assert normalize_tool_name("读取文件") == "local_file.read"
        assert normalize_tool_name("查看文件") == "local_file.read"
        assert normalize_tool_name("read_file") == "local_file.read"

    def test_local_file_list_aliases(self):
        assert normalize_tool_name("列出文件") == "local_file.list"
        assert normalize_tool_name("list_files") == "local_file.list"
        assert normalize_tool_name("ls") == "local_file.list"
        assert normalize_tool_name("查看目录") == "local_file.list"

    def test_local_file_append_aliases(self):
        assert normalize_tool_name("追加文件") == "local_file.append"
        assert normalize_tool_name("追加写入") == "local_file.append"
        assert normalize_tool_name("追加") == "local_file.append"
        assert normalize_tool_name("append_file") == "local_file.append"

    def test_local_file_delete_aliases(self):
        assert normalize_tool_name("删除文件") == "local_file.delete"
        assert normalize_tool_name("delete_file") == "local_file.delete"

    def test_email_draft_aliases(self):
        assert normalize_tool_name("草稿") == "email_mcp.create_draft"
        assert normalize_tool_name("email_draft") == "email_mcp.create_draft"

    def test_unknown_name_passthrough(self):
        assert normalize_tool_name("nonexistent_tool") == "nonexistent_tool"
        assert normalize_tool_name("不存在的工具") == "不存在的工具"

    def test_empty_name(self):
        assert normalize_tool_name("") == ""

    def test_canonical_name_maps_to_itself(self):
        assert normalize_tool_name("email.send") == "email.send"
        assert normalize_tool_name("local_file.write") == "local_file.write"
        assert normalize_tool_name("local_file.list") == "local_file.list"

    def test_normalize_when_canonical_not_in_alias_map(self):
        """Tools without explicit aliases passthrough canonical name unchanged."""
        assert normalize_tool_name("search_mcp.search") == "search_mcp.search"
        assert normalize_tool_name("skill_mcp.create_draft") == "skill_mcp.create_draft"
