from fastapi.testclient import TestClient

from src.web_app.main import app


def test_health_dependencies_no_crash():
    response = TestClient(app).get("/api/v1/health/dependencies")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "database" in data["data"]
