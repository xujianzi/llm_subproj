"""web/backend/services/map_service.py — ACS query + GeoJSON join + stats."""
from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root is in sys.path (injected by main.py / test setup)
from query_db import get_column_names, query_acs_data

_GEODATA = Path(__file__).resolve().parent.parent / "geodata"
_DATA    = Path(__file__).resolve().parent.parent / "data"

# Cache GeoJSON files in memory (loaded once per process)
_gj_cache: dict[str, dict] = {}


def _load_geojson(name: str) -> dict:
    if name not in _gj_cache:
        path = _GEODATA / name
        _gj_cache[name] = json.loads(path.read_text(encoding="utf-8"))
    return _gj_cache[name]


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(rows: List[Dict], variable: str) -> Optional[Dict]:
    """Compute min/max/mean/median for a numeric variable across rows."""
    values = [r[variable] for r in rows if r.get(variable) is not None]
    if not values:
        return None
    return {
        "variable": variable,
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


# ── GeoJSON Join ───────────────────────────────────────────────────────────────

def join_geojson(geojson: dict, rows: List[Dict], level: str) -> dict:
    """Merge ACS rows into GeoJSON Feature.properties by join key."""
    gj = copy.deepcopy(geojson)

    if level == "state":
        lookup = {r["state"].upper(): r for r in rows if r.get("state")}
        for feat in gj["features"]:
            key = feat["properties"].get("STUSPS", "").upper()
            if key in lookup:
                feat["properties"].update(lookup[key])

    elif level == "county":
        lookup = {
            (r["state"].upper(), r["county"].lower()): r
            for r in rows
            if r.get("state") and r.get("county")
        }
        for feat in gj["features"]:
            p = feat["properties"]
            key = (p.get("STUSPS", "").upper(), p.get("NAME", "").lower())
            if key in lookup:
                feat["properties"].update(lookup[key])

    elif level == "zipcode":
        lookup = {str(r["zipcode"]).zfill(5): r for r in rows if r.get("zipcode")}
        for feat in gj["features"]:
            key = feat["properties"].get("ZCTA5CE20", "")
            if key in lookup:
                feat["properties"].update(lookup[key])

    return gj


# ── Public API ─────────────────────────────────────────────────────────────────

def get_map_data(
    level: str,
    variables: List[str],
    year: int,
    state: Optional[str] = None,
    county: Optional[str] = None,
) -> Dict[str, Any]:
    """Query ACS, load GeoJSON boundary, join, compute stats."""
    rows = query_acs_data(
        variables=variables,
        state=state,
        county=county,
        year=year,
    )

    if level == "state":
        geojson = _load_geojson("states.geojson")
    elif level == "county":
        geojson = _load_geojson("counties.geojson")
    else:  # zipcode
        geojson = _load_geojson("zcta_all.geojson")

    # Filter county GeoJSON to selected state
    if level == "county" and state:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                f for f in geojson["features"]
                if f["properties"].get("STUSPS", "").upper() == state.upper()
            ],
        }

    joined = join_geojson(geojson, rows, level=level)
    stats = compute_stats(rows, variables[0]) if variables else None

    return {"geojson": joined, "stats": stats}


def get_variables() -> List[str]:
    return get_column_names()


def get_regions(level: str, state: Optional[str] = None) -> List[str]:
    county_data = json.loads((_DATA / "county_names.json").read_text(encoding="utf-8"))
    if level == "county" and state:
        return county_data.get(state.upper(), [])
    return []
