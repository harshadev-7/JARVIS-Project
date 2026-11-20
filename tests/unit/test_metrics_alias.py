from fastapi.testclient import TestClient

from api.app import app


def test_metrics_alias():
    client = TestClient(app)
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "cpu_percent" in data
    assert "ram_total_gb" in data
    assert "disk_total_gb" in data
