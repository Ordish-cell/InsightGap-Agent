from src.web_app.api.v1.router import api_router


def test_router_imports():
    assert api_router.routes
