from fastapi.testclient import TestClient

from api.app import app


def test_ready_alias():
    client = TestClient(app)
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert "version" in data
