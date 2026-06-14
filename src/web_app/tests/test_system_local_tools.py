from src.web_app.mcp.local_provider import local_provider
from src.web_app.mcp.tool_router import infer_tool


def test_infer_today_date_uses_system_time():
    tool_name, tool_input = infer_tool("今天是几月几号？", {})

    assert tool_name == "system.time"
    assert tool_input == {}


def test_infer_claude_latest_model_uses_rewritten_web_search_query():
    tool_name, tool_input = infer_tool("claude的最新模型是啥？？", {})

    assert tool_name == "web.search"
    assert tool_input["query"] == "site:anthropic.com/news Claude Opus latest model"


def test_infer_generic_latest_model_uses_official_query():
    tool_name, tool_input = infer_tool("Gemini 的最新模型是什么？", {})

    assert tool_name == "web.search"
    assert tool_input["query"] == "Gemini latest model official docs announcement"


def test_infer_latest_version_uses_official_query_without_recency_filter():
    tool_name, tool_input = infer_tool("Next.js 最新版本是多少？", {})

    assert tool_name == "web.search"
    assert tool_input["query"] == "Next.js latest version official docs release notes"
    assert "recency_days" not in tool_input


def test_infer_calc_uses_system_calc():
    tool_name, tool_input = infer_tool("计算 12345 × 6789", {})

    assert tool_name == "system.calc"
    assert tool_input["expression"] == "12345 * 6789"


def test_system_time_returns_local_clock_fields():
    result = local_provider.call(None, 1, "system.time", {}, None)  # type: ignore[arg-type]

    assert result["date"]
    assert result["time"]
    assert result["weekday"]
    assert result["iso"]


def test_system_calc_returns_result():
    result = local_provider.call(None, 1, "system.calc", {"expression": "12 * (5 + 3)"}, None)  # type: ignore[arg-type]

    assert result["result"] == 96


def test_system_unit_convert_returns_result():
    result = local_provider.call(None, 1, "system.unit_convert", {"value": 100, "from": "km", "to": "mile"}, None)  # type: ignore[arg-type]

    assert result["result"] == 62.137119


def test_system_hash_returns_hash():
    result = local_provider.call(None, 1, "system.hash", {"algorithm": "sha256", "text": "hello"}, None)  # type: ignore[arg-type]

    assert result["hash"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
