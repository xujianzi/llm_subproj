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


# ── get_variables ─────────────────────────────────────────────────────────────

def test_get_variables_returns_list():
    from services.map_service import get_variables
    with patch("services.map_service.get_column_names", return_value=["population", "median_income"]):
        result = get_variables()
    assert "population" in result
    assert isinstance(result, list)


# ── get_regions ───────────────────────────────────────────────────────────────

def test_get_regions_county_returns_list(tmp_path):
    import json
    from services.map_service import get_regions
    # Patch _DATA path to use tmp_path
    mock_data = {"NY": ["Kings", "Queens"], "CA": ["Los Angeles"]}
    county_file = tmp_path / "county_names.json"
    county_file.write_text(json.dumps(mock_data), encoding="utf-8")

    import services.map_service as ms
    original_data = ms._DATA
    ms._DATA = tmp_path
    try:
        result = get_regions("county", "NY")
        assert "Kings" in result
        assert "Queens" in result
    finally:
        ms._DATA = original_data

def test_get_regions_non_county_returns_empty(tmp_path):
    import json
    from services.map_service import get_regions
    import services.map_service as ms
    mock_data = {"NY": ["Kings"]}
    (tmp_path / "county_names.json").write_text(json.dumps(mock_data), encoding="utf-8")
    original_data = ms._DATA
    ms._DATA = tmp_path
    try:
        result = get_regions("state", "NY")
        assert result == []
    finally:
        ms._DATA = original_data


# ── get_map_data ──────────────────────────────────────────────────────────────

def test_get_map_data_state_returns_geojson_and_stats():
    from services.map_service import get_map_data
    mock_rows = [{"state": "NY", "median_income": 65000}, {"state": "CA", "median_income": 75000}]
    mock_gj = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"STUSPS": "NY", "NAME": "New York"}, "geometry": None},
            {"type": "Feature", "properties": {"STUSPS": "CA", "NAME": "California"}, "geometry": None},
        ]
    }
    with patch("services.map_service.query_acs_data", return_value=mock_rows), \
         patch("services.map_service._load_geojson", return_value=mock_gj):
        result = get_map_data("state", ["median_income"], 2020)
    assert "geojson" in result
    assert "stats" in result
    assert result["stats"]["variable"] == "median_income"
    assert result["stats"]["min"] == 65000
    ny = next(f for f in result["geojson"]["features"] if f["properties"]["STUSPS"] == "NY")
    assert ny["properties"]["median_income"] == 65000


def test_get_map_data_county_filters_by_state():
    from services.map_service import get_map_data
    mock_rows = [{"state": "NY", "county": "Kings", "median_income": 50000}]
    mock_gj = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"STUSPS": "NY", "NAME": "Kings",  "STATEFP": "36"}, "geometry": None},
            {"type": "Feature", "properties": {"STUSPS": "CA", "NAME": "Alameda", "STATEFP": "06"}, "geometry": None},
        ]
    }
    with patch("services.map_service.query_acs_data", return_value=mock_rows), \
         patch("services.map_service._load_geojson", return_value=mock_gj):
        result = get_map_data("county", ["median_income"], 2020, state="NY")
    # County filter should keep only NY features
    states_in_result = {f["properties"]["STUSPS"] for f in result["geojson"]["features"]}
    assert "CA" not in states_in_result


def test_get_map_data_empty_variables_stats_is_none():
    from services.map_service import get_map_data
    mock_gj = {"type": "FeatureCollection", "features": []}
    with patch("services.map_service.query_acs_data", return_value=[]), \
         patch("services.map_service._load_geojson", return_value=mock_gj):
        result = get_map_data("state", [], 2020)
    assert result["stats"] is None
