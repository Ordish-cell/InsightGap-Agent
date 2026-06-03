from src.web_app.context.builder import ContextBuilder


def test_context_builder_sections():
    context = ContextBuilder().build({"task": "answer", "evidence": [{"title": "source"}]})
    assert "[Role & Policies]" in context
    assert "[Evidence]" in context
