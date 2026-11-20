from fastapi.testclient import TestClient

from api.app import app


def test_root_health_alias():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert "version" in data
    assert "uptime_seconds" in data


def test_status_alias():
    client = TestClient(app)
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert "provider" in data
