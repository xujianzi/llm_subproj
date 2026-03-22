from fastapi.testclient import TestClient
from main import app


def test_status():
    client = TestClient(app)
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
