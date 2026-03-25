# web/tests/backend/test_map_router.py
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent.parent / "backend"
_root    = Path(__file__).resolve().parent.parent.parent.parent
for p in [str(_backend), str(_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest.mock import patch
from fastapi.testclient import TestClient

# Import app as a package (relative imports require this)
_web = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_web))
from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_variables():
    with patch("backend.routers.map_router.get_variables", return_value=["population", "median_income"]):
        r = client.get("/api/map/variables")
    assert r.status_code == 200
    assert "population" in r.json()["columns"]


def test_get_regions_county():
    with patch("backend.routers.map_router.get_regions", return_value=["Kings", "Queens"]):
        r = client.get("/api/map/regions?level=county&state=NY")
    assert r.status_code == 200
    assert "Kings" in r.json()["regions"]


def test_get_map_data_state():
    mock_result = {
        "geojson": {"type": "FeatureCollection", "features": []},
        "stats": {"variable": "median_income", "min": 30000, "max": 90000, "mean": 60000, "median": 58000},
    }
    with patch("backend.routers.map_router.get_map_data", return_value=mock_result):
        r = client.get("/api/map/data?level=state&variables=median_income&year=2020")
    assert r.status_code == 200
    body = r.json()
    assert "geojson" in body
    assert body["stats"]["variable"] == "median_income"


def test_get_map_data_county_missing_state_returns_422():
    r = client.get("/api/map/data?level=county&variables=median_income&year=2020")
    assert r.status_code == 422
