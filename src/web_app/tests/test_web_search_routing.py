from src.web_app.agent.runtime.planner import plan_route


def test_today_date_routes_to_system_time_not_web_search():
    plan = plan_route("今天是几月几号？")

    assert plan["intent"] == "system.time"
    assert plan["route"] == ["tool_agent", "evaluator", "final_response"]
    assert plan["risk_level"] == "L0"


def test_now_time_routes_to_system_time_not_web_search():
    plan = plan_route("现在几点？")

    assert plan["intent"] == "system.time"
    assert "tool_agent" in plan["route"]
    assert plan["expected_output"] == "local_tool_result"


def test_today_weekday_routes_to_system_time_not_web_search():
    plan = plan_route("今天星期几？")

    assert plan["intent"] == "system.time"
    assert "tool_agent" in plan["route"]


def test_latest_current_query_routes_to_light_web_search():
    plan = plan_route("查一下今天 OpenAI 有什么最新消息")

    assert plan["intent"] == "tool.web_search"
    assert plan["route"] == ["tool_agent", "evaluator", "final_response"]
    assert plan["risk_level"] == "L1"
    assert plan["research_mode"] == "none"


def test_stock_price_routes_to_light_web_search():
    plan = plan_route("现在英伟达股价多少？")

    assert plan["intent"] == "tool.web_search"
    assert "tool_agent" in plan["route"]


def test_current_version_routes_to_light_web_search():
    plan = plan_route("当前 Next.js 最新版本是多少？")

    assert plan["intent"] == "tool.web_search"
    assert "tool_agent" in plan["route"]


def test_latest_model_routes_to_light_web_search():
    plan = plan_route("claude的最新模型是啥？？")

    assert plan["intent"] == "tool.web_search"
    assert "tool_agent" in plan["route"]


def test_weather_routes_to_light_web_search():
    plan = plan_route("今天北京天气怎么样？")

    assert plan["intent"] == "tool.web_search"
    assert "tool_agent" in plan["route"]


def test_deep_research_stays_on_odr_route():
    plan = plan_route("帮我深度调研 OpenAI Agent 趋势")

    assert plan["intent"] == "research"
    assert "research_agent" in plan["route"]
    assert "tool_agent" not in plan["route"]
    assert plan["research_mode"] == "deep"


def test_project_advice_does_not_trigger_web_search():
    plan = plan_route("这个项目用 FastAPI 怎么设计架构")

    assert plan["intent"] == "chat"
    assert "tool_agent" not in plan["route"]
    assert "research_agent" not in plan["route"]


def test_document_question_stays_rag_not_web_search():
    plan = plan_route("根据我上传的文档回答", has_document_attachments=True)

    assert plan["intent"] in {"rag", "document_qa"}
    assert "rag_agent" in plan["route"]
    assert "tool_agent" not in plan["route"]


def test_short_general_question_with_attachment_does_not_force_rag():
    plan = plan_route("今年谁获得了NBA总冠军？", has_document_attachments=True)

    assert plan["intent"] != "document_qa"
    assert "rag_agent" not in plan["route"]
    assert "short_query_with_attachment" not in plan["reason"]


def test_english_general_question_with_attachment_does_not_force_rag():
    plan = plan_route("Who won this year?", has_document_attachments=True)

    assert plan["intent"] != "document_qa"
    assert "rag_agent" not in plan["route"]


def test_document_word_alone_with_attachment_does_not_force_rag():
    plan = plan_route("What is a document database?", has_document_attachments=True)

    assert plan["intent"] != "document_qa"
    assert "rag_agent" not in plan["route"]


def test_chinese_document_word_alone_with_attachment_does_not_force_rag():
    plan = plan_route("文档数据库是什么？", has_document_attachments=True)

    assert plan["intent"] != "document_qa"
    assert "rag_agent" not in plan["route"]


def test_explicit_document_reference_with_attachment_routes_to_rag():
    plan = plan_route("根据我上传的文档回答", has_document_attachments=True)

    assert plan["intent"] == "document_qa"
    assert "rag_agent" in plan["route"]
    assert "research_agent" not in plan["route"]
    assert "artifact_agent" not in plan["route"]
    assert "tool_agent" not in plan["route"]


def test_memory_write_does_not_trigger_web_search():
    plan = plan_route("记住我喜欢 Tavily")

    assert plan["intent"] == "memory"
    assert "memory_agent" in plan["route"]
    assert "tool_agent" not in plan["route"]


def test_writing_request_with_today_does_not_trigger_web_search():
    plan = plan_route("帮我写一封今天发给客户的邮件")

    assert plan["intent"] != "tool.web_search"
