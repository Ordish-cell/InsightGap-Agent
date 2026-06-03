from src.web_app.context.builder import ContextBuilder


def build_context(payload: dict) -> str:
    return ContextBuilder().build(payload)
