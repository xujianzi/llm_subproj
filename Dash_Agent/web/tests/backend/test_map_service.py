# web/tests/backend/test_map_service.py
import sys
from pathlib import Path

# Adjust sys.path for both the backend and repo root
_backend = Path(__file__).resolve().parent.parent.parent / "backend"
_root    = Path(__file__).resolve().parent.parent.parent.parent
for p in [str(_backend), str(_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from unittest.mock import patch
from services.map_service import compute_stats, join_geojson


# ── compute_stats ─────────────────────────────────────────────────────────────

def test_compute_stats_basic():
    rows = [{"median_income": 40000}, {"median_income": 60000}, {"median_income": 80000}]
    stats = compute_stats(rows, "median_income")
    assert stats["min"] == 40000
    assert stats["max"] == 80000
    assert stats["mean"] == pytest.approx(60000)
    assert stats["median"] == 60000
    assert stats["variable"] == "median_income"

def test_compute_stats_skips_none():
    rows = [{"median_income": None}, {"median_income": 50000}, {"median_income": 100000}]
    stats = compute_stats(rows, "median_income")
    assert stats["min"] == 50000
    assert stats["max"] == 100000

def test_compute_stats_empty():
    stats = compute_stats([], "median_income")
    assert stats is None


# ── join_geojson — state level ────────────────────────────────────────────────

MOCK_STATE_GJ = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"STUSPS": "NY", "NAME": "New York"}, "geometry": None},
        {"type": "Feature", "properties": {"STUSPS": "CA", "NAME": "California"}, "geometry": None},
    ]
}

def test_join_geojson_state_level():
    rows = [{"state": "NY", "median_income": 65000}, {"state": "CA", "median_income": 75000}]
    result = join_geojson(MOCK_STATE_GJ, rows, level="state")
    features = result["features"]
    ny = next(f for f in features if f["properties"]["STUSPS"] == "NY")
    assert ny["properties"]["median_income"] == 65000

def test_join_geojson_no_match_keeps_feature():
    rows = [{"state": "NY", "median_income": 65000}]
    result = join_geojson(MOCK_STATE_GJ, rows, level="state")
    ca = next(f for f in result["features"] if f["properties"]["STUSPS"] == "CA")
    assert "median_income" not in ca["properties"]


# ── join_geojson — county level ───────────────────────────────────────────────

MOCK_COUNTY_GJ = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"STUSPS": "NY", "NAME": "Kings",  "STATEFP": "36"}, "geometry": None},
        {"type": "Feature", "properties": {"STUSPS": "NY", "NAME": "Queens", "STATEFP": "36"}, "geometry": None},
    ]
}

def test_join_geojson_county_level():
    rows = [
        {"state": "NY", "county": "Kings",  "median_income": 50000},
        {"state": "NY", "county": "Queens", "median_income": 55000},
    ]
    result = join_geojson(MOCK_COUNTY_GJ, rows, level="county")
    kings = next(f for f in result["features"] if f["properties"]["NAME"] == "Kings")
    assert kings["properties"]["median_income"] == 50000

def test_join_geojson_county_case_insensitive():
    rows = [{"state": "ny", "county": "KINGS", "median_income": 50000}]
    result = join_geojson(MOCK_COUNTY_GJ, rows, level="county")
    kings = next(f for f in result["features"] if f["properties"]["NAME"] == "Kings")
    assert kings["properties"]["median_income"] == 50000


# ── join_geojson — zipcode level ──────────────────────────────────────────────

MOCK_ZCTA_GJ = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"ZCTA5CE20": "10001"}, "geometry": None},
        {"type": "Feature", "properties": {"ZCTA5CE20": "10002"}, "geometry": None},
    ]
}

def test_join_geojson_zipcode_level():
    rows = [{"zipcode": "10001", "median_income": 70000}]
    result = join_geojson(MOCK_ZCTA_GJ, rows, level="zipcode")
    zcta = next(f for f in result["features"] if f["properties"]["ZCTA5CE20"] == "10001")
    assert zcta["properties"]["median_income"] == 70000
