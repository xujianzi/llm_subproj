# ACS Data Explorer Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + React web app that visualizes ACS data on a Mapbox choropleth map and lets users query it via a floating Chat Agent panel backed by the existing LLM agent.

**Architecture:** FastAPI backend in `web/backend/` exposes REST + SSE endpoints; React frontend in `web/frontend/` renders a full-screen Mapbox map with a floating ConfigPanel, StatsCard, and ChatPanel. All new code lives in `web/`—no existing files are modified. The backend imports from parent-directory modules (`agent.py`, `query_db.py`, etc.) via `sys.path` injection.

**Tech Stack:** Python/FastAPI/uvicorn/sse-starlette (backend) · React 18/Vite/TypeScript/Tailwind/shadcn-ui/Mapbox GL JS/Zustand/AG Grid (frontend)

**Spec:** `docs/superpowers/specs/2026-03-25-acs-web-app-design.md`

---

## Prerequisites (manual, one-time)

Before running the plan, complete these setup steps:

1. **Mapbox Token** — sign up at mapbox.com (free tier), copy your public token.
2. **GeoJSON boundary files** — download Census cartographic shapefiles and convert to GeoJSON. See Task 1 for the exact commands.
3. **Python venv** — the existing venv at `Dash_Agent/` root already has psycopg2, openai, pydantic-settings. Backend `requirements.txt` adds FastAPI etc. to the same venv.

---

## File Map

```
web/
├── backend/
│   ├── main.py                     NEW — FastAPI app entry point
│   ├── requirements.txt            NEW — backend Python deps
│   ├── routers/
│   │   ├── __init__.py             NEW — empty
│   │   ├── map_router.py           NEW — /api/map/* routes
│   │   └── chat_router.py          NEW — /api/chat/stream route
│   ├── services/
│   │   ├── __init__.py             NEW — empty
│   │   ├── map_service.py          NEW — ACS query + GeoJSON join + stats
│   │   └── chat_service.py         NEW — per-request agent loop + SSE generator
│   ├── geodata/
│   │   ├── states.geojson          NEW — ~0.5 MB, 56 features (STUSPS + NAME)
│   │   ├── counties.geojson        NEW — ~3 MB, ~3200 features
│   │   └── zcta_all.geojson        NEW — ~30-80 MB, all ZCTAs (500K resolution, already small)
│   └── data/
│       ├── state_fips.json         NEW — {"NY":"36","CA":"06",...}
│       └── county_names.json       NEW — {"NY":["Kings","Queens",...],...}
│
├── frontend/
│   ├── .env.local                  NEW — VITE_MAPBOX_TOKEN=... (not committed)
│   ├── .gitignore                  NEW
│   ├── index.html                  NEW — Vite entry HTML
│   ├── package.json                NEW
│   ├── tsconfig.json               NEW
│   ├── vite.config.ts              NEW — proxy /api → :8000
│   ├── tailwind.config.ts          NEW
│   └── src/
│       ├── main.tsx                NEW — React root mount
│       ├── App.tsx                 NEW — tab routing + layout shell
│       ├── store/
│       │   └── useMapStore.ts      NEW — Zustand store (all shared state)
│       ├── api/
│       │   ├── mapApi.ts           NEW — fetch wrappers for /api/map/*
│       │   └── chatApi.ts          NEW — SSE consumer for /api/chat/stream
│       ├── components/
│       │   ├── MapView/
│       │   │   └── MapView.tsx     NEW — Mapbox GL + choropleth + tooltip
│       │   ├── ConfigPanel/
│       │   │   └── ConfigPanel.tsx NEW — Year/Variable/State/County dropdowns
│       │   ├── ChatPanel/
│       │   │   └── ChatPanel.tsx   NEW — floating chat window + SSE
│       │   ├── StatsCard/
│       │   │   └── StatsCard.tsx   NEW — floating min/max/mean/median
│       │   └── DataTable/
│       │       └── DataTable.tsx   NEW — AG Grid tab view
│       └── types/
│           └── index.ts            NEW — shared TypeScript types
│
└── tests/
    └── backend/
        ├── test_map_service.py     NEW — unit tests for join + stats logic
        └── test_map_router.py      NEW — FastAPI TestClient integration tests
```

---

## Task 1: Geodata — Download and Process Boundary Files

**Files:**
- Create: `web/backend/geodata/states.geojson`
- Create: `web/backend/geodata/counties.geojson`
- Create: `web/backend/geodata/zcta/<state>.geojson` (for each state needed)
- Create: `web/backend/data/state_fips.json`
- Create: `web/backend/data/county_names.json`

> This task is a one-time data preparation step. It does not involve application code.

- [ ] **Step 1: Create geodata directory structure**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent
mkdir -p web/backend/geodata/zcta
mkdir -p web/backend/data
```

- [ ] **Step 2: Install ogr2ogr (if not present)**

ogr2ogr is part of GDAL. On Windows with conda:
```bash
conda install -c conda-forge gdal
```
Or download OSGeo4W from osgeo.org. Verify: `ogr2ogr --version`

- [ ] **Step 3: Download Census cartographic boundary shapefiles**

> Note: On Windows use `./tmp/` as temp directory (not `/tmp/`). Run these commands in Git Bash.

```bash
mkdir -p ./tmp

# State boundaries (20m resolution, ~1MB)
curl -L "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip" -o ./tmp/states.zip
unzip ./tmp/states.zip -d ./tmp/states_shp

# County boundaries (20m resolution, ~10MB)
curl -L "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip" -o ./tmp/counties.zip
unzip ./tmp/counties.zip -d ./tmp/counties_shp

# ZCTA 500K cartographic boundary — this is the SMALL version (~5MB zip, ~30-80MB GeoJSON)
# 500K resolution is pre-simplified; no need for per-state splitting.
curl -L "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_zcta520_500k.zip" -o ./tmp/zcta.zip
unzip ./tmp/zcta.zip -d ./tmp/zcta_shp
```

- [ ] **Step 4: Convert shapefiles to GeoJSON**

```bash
# States: keep STUSPS (abbrev) and NAME fields
ogr2ogr -f GeoJSON \
  -select "STUSPS,NAME" \
  web/backend/geodata/states.geojson \
  ./tmp/states_shp/cb_2022_us_state_20m.shp

# Counties: keep STUSPS, NAME, STATEFP fields
ogr2ogr -f GeoJSON \
  -select "STUSPS,NAME,STATEFP" \
  web/backend/geodata/counties.geojson \
  ./tmp/counties_shp/cb_2022_us_county_20m.shp

# ZCTA: keep ZCTA5CE20 only — single file (500K resolution = already simplified)
ogr2ogr -f GeoJSON \
  -select "ZCTA5CE20" \
  web/backend/geodata/zcta_all.geojson \
  ./tmp/zcta_shp/cb_2022_us_zcta520_500k.shp
```

- [ ] **Step 5: Build static data files (county_names.json + state_fips.json)**

Create `web/backend/data/prepare_data.py`:

```python
"""One-time script: build county_names.json and state_fips.json from counties GeoJSON."""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
GEODATA = ROOT.parent / "geodata"

# Load counties GeoJSON (already converted in Step 4)
with open(GEODATA / "counties.geojson", encoding="utf-8") as f:
    counties = json.load(f)

# ── Build county_names.json ──────────────────────────────────────────────────
county_names: dict = defaultdict(list)
fips_to_abbr: dict = {}

for feat in counties["features"]:
    p = feat["properties"]
    state_abbr = p.get("STUSPS", "")
    county_name = p.get("NAME", "")
    statefp = p.get("STATEFP", "")
    if state_abbr and county_name:
        county_names[state_abbr].append(county_name)
    if statefp and state_abbr:
        fips_to_abbr[statefp] = state_abbr

for state in county_names:
    county_names[state] = sorted(county_names[state])

out = ROOT / "county_names.json"
out.write_text(json.dumps(dict(county_names), indent=2), encoding="utf-8")
print(f"Wrote county_names.json: {len(county_names)} states")

# ── Build state_fips.json ────────────────────────────────────────────────────
state_fips = {abbr: fips for fips, abbr in fips_to_abbr.items()}
out = ROOT / "state_fips.json"
out.write_text(json.dumps(state_fips, indent=2), encoding="utf-8")
print(f"Wrote state_fips.json: {len(state_fips)} entries")
```

Run:
```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent
python web/backend/data/prepare_data.py
```
Expected output:
```
Wrote county_names.json: 56 states
Wrote state_fips.json: 56 entries
```

> The `zcta_all.geojson` file was already written directly by ogr2ogr in Step 4 — no Python processing needed.

- [ ] **Step 6: Verify files exist and are valid JSON**

```bash
python -c "
import json
from pathlib import Path
base = Path('web/backend')
for p in [
    base/'geodata/states.geojson',
    base/'geodata/counties.geojson',
    base/'data/state_fips.json',
    base/'data/county_names.json',
]:
    data = json.loads(p.read_text())
    print(p.name, '✓')
"
```

- [ ] **Step 7: Commit geodata and static data**

```bash
git add web/backend/geodata/ web/backend/data/
git commit -m "feat: add Census boundary GeoJSON and static lookup data"
```

---

## Task 2: Backend Scaffold

**Files:**
- Create: `web/backend/requirements.txt`
- Create: `web/backend/main.py`
- Create: `web/backend/routers/__init__.py`
- Create: `web/backend/services/__init__.py`
- Create: `web/tests/backend/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
web/backend/requirements.txt
─────────────────────────────
fastapi>=0.111
uvicorn[standard]>=0.29
sse-starlette>=1.8
psycopg2-binary>=2.9
pydantic-settings>=2.2
python-dotenv>=1.0
openai>=1.30
pytest>=8
httpx>=0.27
```

Install into the existing venv:
```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent
pip install -r web/backend/requirements.txt
```

- [ ] **Step 2: Create directory structure and stub router files**

```bash
mkdir -p web/backend/routers web/backend/services web/tests/backend
touch web/backend/routers/__init__.py
touch web/backend/services/__init__.py
touch web/tests/__init__.py
touch web/tests/backend/__init__.py
```

Create stub routers so `main.py` can import them immediately (filled in Tasks 4 and 6):

```python
# web/backend/routers/map_router.py  (stub)
from fastapi import APIRouter
router = APIRouter()
```

```python
# web/backend/routers/chat_router.py  (stub)
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 3: Create `web/backend/main.py`**

```python
"""web/backend/main.py — FastAPI application entry point."""
import sys
from pathlib import Path

# Inject parent repo root so we can import agent.py, query_db.py, etc.
_root = str(Path(__file__).parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.map_router import router as map_router
from routers.chat_router import router as chat_router

app = FastAPI(title="ACS Data Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(map_router, prefix="/api/map")
app.include_router(chat_router, prefix="/api/chat")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Verify the app starts**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent/web/backend
uvicorn main:app --reload --port 8000
```
Expected: `Application startup complete.` (Ctrl+C to stop)

- [ ] **Step 5: Commit scaffold**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent
git add web/backend/requirements.txt web/backend/main.py \
        web/backend/routers/__init__.py web/backend/services/__init__.py \
        web/tests/
git commit -m "feat: add FastAPI backend scaffold with CORS and health endpoint"
```

---

## Task 3: map_service.py — ACS Query + GeoJSON Join + Stats

**Files:**
- Create: `web/backend/services/map_service.py`
- Create: `web/tests/backend/test_map_service.py`

- [ ] **Step 1: Write the failing tests**

```python
# web/tests/backend/test_map_service.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

import json
import pytest
from unittest.mock import patch
from services.map_service import compute_stats, join_geojson


# ── compute_stats ──────────────────────────────────────────────────────────

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


# ── join_geojson ─────────────────────────────────────────────────────────

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
    # CA has no ACS row — feature still present, no ACS property
    ca = next(f for f in result["features"] if f["properties"]["STUSPS"] == "CA")
    assert "median_income" not in ca["properties"]


MOCK_COUNTY_GJ = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"STUSPS": "NY", "NAME": "Kings",   "STATEFP": "36"}, "geometry": None},
        {"type": "Feature", "properties": {"STUSPS": "NY", "NAME": "Queens",  "STATEFP": "36"}, "geometry": None},
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent
pytest web/tests/backend/test_map_service.py -v
```
Expected: `ImportError: cannot import name 'compute_stats' from 'services.map_service'`

- [ ] **Step 3: Implement `web/backend/services/map_service.py`**

```python
"""web/backend/services/map_service.py — ACS query + GeoJSON join + stats."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root is in sys.path (injected by main.py)
from query_db import get_column_names, query_acs_data

_GEODATA = Path(__file__).parent.parent / "geodata"
_DATA    = Path(__file__).parent.parent / "data"

# Cache GeoJSON files in memory (loaded once per process)
_gj_cache: dict[str, dict] = {}


def _load_geojson(name: str) -> dict:
    if name not in _gj_cache:
        path = _GEODATA / name
        _gj_cache[name] = json.loads(path.read_text(encoding="utf-8"))
    return _gj_cache[name]


# ── Stats ─────────────────────────────────────────────────────────────────

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
        "median": statistics.median(values),
    }


# ── GeoJSON Join ──────────────────────────────────────────────────────────

def join_geojson(geojson: dict, rows: List[Dict], level: str) -> dict:
    """Merge ACS rows into GeoJSON Feature.properties by join key."""
    import copy
    gj = copy.deepcopy(geojson)

    if level == "state":
        # Key: STUSPS (e.g. "NY")
        lookup = {r["state"].upper(): r for r in rows if r.get("state")}
        for feat in gj["features"]:
            key = feat["properties"].get("STUSPS", "").upper()
            if key in lookup:
                feat["properties"].update(lookup[key])

    elif level == "county":
        # Key: (state_abbr, county_name) case-insensitive
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
        # Key: ZCTA5CE20 (5-digit string)
        lookup = {str(r["zipcode"]).zfill(5): r for r in rows if r.get("zipcode")}
        for feat in gj["features"]:
            key = feat["properties"].get("ZCTA5CE20", "")
            if key in lookup:
                feat["properties"].update(lookup[key])

    return gj


# ── Public API ────────────────────────────────────────────────────────────

def get_map_data(
    level: str,
    variables: List[str],
    year: int,
    state: Optional[str] = None,
    county: Optional[str] = None,
) -> Dict[str, Any]:
    """Query ACS, load GeoJSON boundary, join, compute stats. Return dict."""
    rows = query_acs_data(
        variables=variables,
        state=state,
        county=county,
        year=year,
    )

    # Load appropriate GeoJSON
    if level == "state":
        geojson = _load_geojson("states.geojson")
    elif level == "county":
        geojson = _load_geojson("counties.geojson")
    else:  # zipcode
        geojson = _load_geojson("zcta_all.geojson")

    # Filter county GeoJSON to selected state to reduce response size
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
    county_data = json.loads((_DATA / "county_names.json").read_text())
    if level == "county" and state:
        return county_data.get(state.upper(), [])
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest web/tests/backend/test_map_service.py -v
```
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add web/backend/services/map_service.py web/tests/backend/test_map_service.py
git commit -m "feat: add map_service with GeoJSON join and stats computation"
```

---

## Task 4: map_router.py — Map API Routes

**Files:**
- Create: `web/backend/routers/map_router.py`
- Create: `web/tests/backend/test_map_router.py`

- [ ] **Step 1: Write failing route tests**

```python
# web/tests/backend/test_map_router.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_variables():
    with patch("routers.map_router.get_variables", return_value=["population", "median_income"]):
        r = client.get("/api/map/variables")
    assert r.status_code == 200
    assert "population" in r.json()["columns"]


def test_get_regions_county():
    with patch("routers.map_router.get_regions", return_value=["Kings", "Queens"]):
        r = client.get("/api/map/regions?level=county&state=NY")
    assert r.status_code == 200
    assert "Kings" in r.json()["regions"]


def test_get_map_data_state():
    mock_result = {
        "geojson": {"type": "FeatureCollection", "features": []},
        "stats": {"variable": "median_income", "min": 30000, "max": 90000, "mean": 60000, "median": 58000},
    }
    with patch("routers.map_router.get_map_data", return_value=mock_result):
        r = client.get("/api/map/data?level=state&variables=median_income&year=2020")
    assert r.status_code == 200
    body = r.json()
    assert "geojson" in body
    assert body["stats"]["variable"] == "median_income"


def test_get_map_data_missing_state_for_county_returns_422():
    r = client.get("/api/map/data?level=county&variables=median_income&year=2020")
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest web/tests/backend/test_map_router.py -v
```
Expected: `ImportError` or route 404 errors.

- [ ] **Step 3: Implement `web/backend/routers/map_router.py`**

```python
"""web/backend/routers/map_router.py — /api/map/* routes."""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from services.map_service import get_map_data, get_variables, get_regions

router = APIRouter()


@router.get("/variables")
def variables():
    return {"columns": get_variables()}


@router.get("/regions")
def regions(
    level: str = Query(...),
    state: Optional[str] = Query(default=None),
):
    return {"regions": get_regions(level=level, state=state)}


@router.get("/data")
def map_data(
    level: str = Query(..., pattern="^(state|county|zipcode)$"),
    variables: str = Query(...),          # comma-separated
    year: int = Query(...),
    state: Optional[str] = Query(default=None),
    county: Optional[str] = Query(default=None),
):
    var_list: List[str] = [v.strip() for v in variables.split(",") if v.strip()]
    if not var_list:
        raise HTTPException(422, "variables must not be empty")
    if level in ("county", "zipcode") and not state:
        raise HTTPException(422, f"state is required when level={level}")
    try:
        result = get_map_data(
            level=level,
            variables=var_list,
            year=year,
            state=state,
            county=county,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest web/tests/backend/test_map_router.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Smoke test with curl**

```bash
# Start server in background
cd I:/LLM_proj/llm_subproj/Dash_Agent/web/backend
uvicorn main:app --port 8000 &

curl "http://localhost:8000/api/map/variables"
# Expected: {"columns": [...]}

curl "http://localhost:8000/api/map/regions?level=county&state=NY"
# Expected: {"regions": ["Albany", "Allegany", ...]}
```

- [ ] **Step 6: Commit**

```bash
git add web/backend/routers/map_router.py web/tests/backend/test_map_router.py
git commit -m "feat: add map_router with /data /variables /regions endpoints"
```

---

## Task 5: chat_service.py — Per-Request Agent Loop + SSE

**Files:**
- Create: `web/backend/services/chat_service.py`

> No automated test for this task — requires live LLM + DB. Manual smoke test below.

- [ ] **Step 1: Implement `web/backend/services/chat_service.py`**

```python
"""web/backend/services/chat_service.py

Runs the ACS agent in a thread, captures QueryACSData results,
and yields SSE events: text | data | error | done.

Key design decisions:
- Does NOT modify global TOOL_HANDLERS in agent.py.
- Builds a per-request mini agent loop that replicates agent.py's dispatch
  logic but uses a local handler map, so concurrent requests are isolated.
- asyncio.Queue bridges the sync thread to the async SSE generator.
"""
from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

# Repo root injected by main.py
from agent import client, TOOLS, SYSTEM, MODEL, micro_compact, auto_compact, estimate_tokens, COMPACT_THRESHOLD
from query_db import query_acs_data, get_column_names
from services.map_service import join_geojson, compute_stats

_SENTINEL = object()  # signals queue is done


def _infer_level(rows: List[Dict]) -> str:
    if rows and rows[0].get("zipcode"):
        return "zipcode"
    if rows and rows[0].get("county"):
        return "county"
    return "state"


def _build_data_payload(rows: List[Dict], columns: List[str]) -> Dict[str, Any]:
    """Build the 'data' SSE payload: rows + GeoJSON + stats."""
    from services.map_service import _load_geojson  # lazy import avoids circular

    level = _infer_level(rows)
    if level == "state":
        raw_gj = _load_geojson("states.geojson")
    elif level == "county":
        raw_gj = _load_geojson("counties.geojson")
    else:
        raw_gj = _load_geojson("zcta_all.geojson")

    geojson = join_geojson(raw_gj, rows, level=level)

    # Pick first numeric-looking column for stats (skip identifiers)
    id_cols = {"zipcode", "city", "county", "state", "year", "geoid"}
    stat_col = next((c for c in columns if c not in id_cols), None)
    stats = compute_stats(rows, stat_col) if stat_col else None

    return {
        "rows": rows,
        "columns": columns,
        "geojson": geojson,
        "stats": stats,
    }


def _run_agent_sync(
    input_messages: List[Dict],
    result_queue: "asyncio.Queue",
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Synchronous agent loop — runs in a thread via asyncio.to_thread."""

    def wrapped_query_acs(**kwargs):
        result = query_acs_data(**kwargs)
        # Determine columns from first row
        cols = list(result[0].keys()) if result else []
        payload = _build_data_payload(result, cols)
        # Put result into the queue (thread-safe via call_soon_threadsafe)
        loop.call_soon_threadsafe(result_queue.put_nowait, payload)
        return result  # return to agent for its internal context

    local_handlers = {
        "QueryACSData":   lambda **kw: wrapped_query_acs(**kw),
        "GetColumnNames": lambda **kw: get_column_names(),
        "Compact":        lambda **kw: "__COMPACT__",  # manual compression trigger
    }

    system_msg = {"role": "system", "content": SYSTEM}

    while True:
        micro_compact(input_messages)
        if estimate_tokens(input_messages) > COMPACT_THRESHOLD:
            input_messages[:] = auto_compact(input_messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg] + input_messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message
        input_messages.append(msg.model_dump(exclude_unset=False))

        if not msg.tool_calls:
            break

        manual_compact = False
        for tc in msg.tool_calls:
            handler = local_handlers.get(tc.function.name)
            try:
                output = (
                    handler(**json.loads(tc.function.arguments))
                    if handler
                    else f"Unknown tool: {tc.function.name}"
                )
            except Exception as e:
                output = f"Error: {e}"

            if output == "__COMPACT__":
                manual_compact = True
                output = "Compressing..."

            input_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(output),
            })

        if manual_compact:
            input_messages[:] = auto_compact(input_messages)

    # Signal done
    loop.call_soon_threadsafe(result_queue.put_nowait, _SENTINEL)


async def chat_stream(
    message: str,
    history: List[Dict[str, str]],
) -> AsyncGenerator[Dict, None]:
    """
    Async generator yielding SSE dicts: {"event": ..., "data": ...}
    Caller (chat_router) wraps these in SSE format.
    """
    input_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
    ] + [{"role": "user", "content": message}]

    result_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    try:
        agent_task = asyncio.to_thread(
            _run_agent_sync, input_messages, result_queue, loop
        )
        agent_future = asyncio.ensure_future(agent_task)

        # Drain queue until sentinel
        while True:
            item = await result_queue.get()
            if item is _SENTINEL:
                break
            # item is a data payload dict
            yield {"event": "data", "data": json.dumps(item)}

        # Wait for agent thread to fully finish
        await agent_future

        # Final text response
        last_content = ""
        if input_messages and input_messages[-1].get("role") == "assistant":
            last_content = input_messages[-1].get("content") or ""
        yield {"event": "text", "data": last_content}

    except Exception as e:
        yield {"event": "error", "data": str(e)}

    finally:
        yield {"event": "done", "data": ""}
```

- [ ] **Step 2: Manual smoke test (requires running DB and LLM key)**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent/web/backend
python -c "
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path('.').parent.parent))
from services.chat_service import chat_stream

async def test():
    async for event in chat_stream('纽约州 2020 年的中位收入', []):
        print(event)

asyncio.run(test())
"
```
Expected: prints `data` event with GeoJSON, then `text` event, then `done`.

- [ ] **Step 3: Commit**

```bash
git add web/backend/services/chat_service.py
git commit -m "feat: add chat_service with per-request agent loop and SSE generator"
```

---

## Task 6: chat_router.py — SSE Endpoint

**Files:**
- Create: `web/backend/routers/chat_router.py`

- [ ] **Step 1: Implement `web/backend/routers/chat_router.py`**

```python
"""web/backend/routers/chat_router.py — POST /api/chat/stream (SSE)."""
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from services.chat_service import chat_stream

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


@router.post("/stream")
async def stream(request: ChatRequest):
    async def generator():
        async for event in chat_stream(
            message=request.message,
            history=[m.model_dump() for m in request.history],
        ):
            yield event

    return EventSourceResponse(generator())
```

- [ ] **Step 2: Smoke test the endpoint**

```bash
# Start server
uvicorn main:app --port 8000 --reload &

# Test with curl (keep --no-buffer for streaming)
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"纽约州 2020 年的 median income","history":[]}'
```
Expected: a stream of `event: data`, `event: text`, `event: done` lines.

- [ ] **Step 3: Commit**

```bash
git add web/backend/routers/chat_router.py
git commit -m "feat: add chat_router SSE endpoint /api/chat/stream"
```

---

## Task 7: Frontend Scaffold

**Files:**
- Create: `web/frontend/package.json`
- Create: `web/frontend/vite.config.ts`
- Create: `web/frontend/tailwind.config.ts`
- Create: `web/frontend/tsconfig.json`
- Create: `web/frontend/index.html`
- Create: `web/frontend/src/main.tsx`
- Create: `web/frontend/src/types/index.ts`
- Create: `web/frontend/.gitignore`

- [ ] **Step 1: Scaffold with Vite**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent/web
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: Install dependencies**

```bash
npm install mapbox-gl zustand ag-grid-react ag-grid-community recharts
npm install -D tailwindcss postcss autoprefixer @types/mapbox-gl
npx tailwindcss init -p --ts
```

Install shadcn/ui:
```bash
npx shadcn-ui@latest init
# Choose: Dark theme, CSS variables, default style
```
> **Important:** `shadcn-ui init` will generate its own `tailwind.config.ts` using CSS variable color names (`background`, `foreground`, `primary`, etc.). After running it, **manually merge** the custom colors from Step 4 into the generated config under `theme.extend.colors` alongside the shadcn variables. Do NOT replace the shadcn config entirely — shadcn components depend on its CSS variable names.

Install individual shadcn components used in this app:
```bash
npx shadcn-ui@latest add select button tabs card badge
```

- [ ] **Step 3: Configure `vite.config.ts`**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 4: Configure `tailwind.config.ts`**

```typescript
import type { Config } from 'tailwindcss'

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#0f1117',
        panel:   '#1a1d27',
        border:  '#2a2d3e',
        primary: '#4f7cff',
        accent:  '#00d4aa',
        text:    '#e2e8f0',
        muted:   '#94a3b8',
      },
    },
  },
  plugins: [],
} satisfies Config
```

- [ ] **Step 5: Create `src/types/index.ts`**

```typescript
export type Level = 'state' | 'county' | 'zipcode'

export interface Stats {
  variable: string
  min: number
  max: number
  mean: number
  median: number
}

export interface MapDataResponse {
  geojson: GeoJSON.FeatureCollection
  stats: Stats | null
}

export interface ChatDataPayload {
  rows: Record<string, unknown>[]
  columns: string[]
  geojson: GeoJSON.FeatureCollection
  stats: Stats | null
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}
```

- [ ] **Step 6: Create `src/main.tsx`**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

- [ ] **Step 7: Create placeholder `src/App.tsx`**

```tsx
export default function App() {
  return <div className="bg-bg min-h-screen text-text flex items-center justify-center">
    ACS Data Explorer — Loading...
  </div>
}
```

- [ ] **Step 8: Add `.gitignore`**

```
node_modules/
dist/
.env.local
```

- [ ] **Step 9: Verify frontend starts**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent/web/frontend
npm run dev
```
Expected: `http://localhost:5173/` shows "ACS Data Explorer — Loading..."

- [ ] **Step 10: Create `.env.local` with your Mapbox token**

```
VITE_MAPBOX_TOKEN=pk.eyJ1...your_token_here
```

- [ ] **Step 11: Commit frontend scaffold**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent
git add web/frontend/
git commit -m "feat: add React+Vite frontend scaffold with Tailwind and shadcn/ui"
```

---

## Task 8: Zustand Store + API Helpers

**Files:**
- Create: `web/frontend/src/store/useMapStore.ts`
- Create: `web/frontend/src/api/mapApi.ts`
- Create: `web/frontend/src/api/chatApi.ts`

- [ ] **Step 1: Create `src/api/mapApi.ts`**

```typescript
import type { Level, MapDataResponse } from '../types'

export async function fetchVariables(): Promise<string[]> {
  const r = await fetch('/api/map/variables')
  const data = await r.json()
  return data.columns as string[]
}

export async function fetchRegions(level: string, state: string): Promise<string[]> {
  const r = await fetch(`/api/map/regions?level=${level}&state=${encodeURIComponent(state)}`)
  const data = await r.json()
  return data.regions as string[]
}

export async function fetchMapData(params: {
  level: Level
  variable: string
  year: number
  state?: string
  county?: string
}): Promise<MapDataResponse> {
  const q = new URLSearchParams({
    level: params.level,
    variables: params.variable,
    year: String(params.year),
    ...(params.state ? { state: params.state } : {}),
    ...(params.county ? { county: params.county } : {}),
  })
  const r = await fetch(`/api/map/data?${q}`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}
```

- [ ] **Step 2: Create `src/api/chatApi.ts`**

```typescript
import type { ChatMessage, ChatDataPayload } from '../types'

export interface ChatStreamCallbacks {
  onText: (text: string) => void
  onData: (payload: ChatDataPayload) => void
  onError: (msg: string) => void
  onDone: () => void
}

export function streamChat(
  message: string,
  history: ChatMessage[],
  callbacks: ChatStreamCallbacks,
): () => void {
  const ctrl = new AbortController()

  fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal: ctrl.signal,
  }).then(async (res) => {
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buf = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop() ?? ''
      for (const part of parts) {
        const eventLine = part.match(/^event:\s*(.+)$/m)?.[1]?.trim()
        const dataLine  = part.match(/^data:\s*(.*)$/m)?.[1]?.trim() ?? ''
        if (eventLine === 'text')  callbacks.onText(dataLine)
        if (eventLine === 'data')  callbacks.onData(JSON.parse(dataLine))
        if (eventLine === 'error') callbacks.onError(dataLine)
        if (eventLine === 'done')  callbacks.onDone()
      }
    }
  }).catch((e) => {
    if (e.name !== 'AbortError') callbacks.onError(String(e))
  })

  return () => ctrl.abort()
}
```

- [ ] **Step 3: Create `src/store/useMapStore.ts`**

```typescript
import { create } from 'zustand'
import type { Level, Stats, ChatMessage, ChatDataPayload } from '../types'
import type { FeatureCollection } from 'geojson'

interface MapStore {
  // Map config
  level: Level
  selectedState: string | null
  selectedCounty: string | null
  selectedVariable: string
  selectedYear: number
  availableVariables: string[]
  availableCounties: string[]

  // Map data
  geojsonData: FeatureCollection | null
  stats: Stats | null

  // Chat
  chatHistory: ChatMessage[]
  chatOpen: boolean

  // Actions
  setLevel: (l: Level) => void
  setSelectedState: (s: string | null) => void
  setSelectedCounty: (c: string | null) => void
  setSelectedVariable: (v: string) => void
  setSelectedYear: (y: number) => void
  setAvailableVariables: (cols: string[]) => void
  setAvailableCounties: (counties: string[]) => void
  setMapData: (gj: FeatureCollection, stats: Stats | null) => void
  updateFromChatData: (payload: ChatDataPayload) => void
  addChatMessage: (msg: ChatMessage) => void
  setChatOpen: (open: boolean) => void
}

export const useMapStore = create<MapStore>((set) => ({
  level: 'state',
  selectedState: null,
  selectedCounty: null,
  selectedVariable: 'median_income',
  selectedYear: 2020,
  availableVariables: [],
  availableCounties: [],
  geojsonData: null,
  stats: null,
  chatHistory: [],
  chatOpen: false,

  setLevel: (level) => set({ level }),
  setSelectedState: (selectedState) => set({ selectedState }),
  setSelectedCounty: (selectedCounty) => set({ selectedCounty }),
  setSelectedVariable: (selectedVariable) => set({ selectedVariable }),
  setSelectedYear: (selectedYear) => set({ selectedYear }),
  setAvailableVariables: (availableVariables) => set({ availableVariables }),
  setAvailableCounties: (availableCounties) => set({ availableCounties }),
  setMapData: (geojsonData, stats) => set({ geojsonData, stats }),
  updateFromChatData: ({ geojson, stats }) =>
    set({ geojsonData: geojson, stats }),
  addChatMessage: (msg) =>
    set((s) => ({ chatHistory: [...s.chatHistory, msg] })),
  setChatOpen: (chatOpen) => set({ chatOpen }),
}))
```

- [ ] **Step 4: Commit**

```bash
git add web/frontend/src/store/ web/frontend/src/api/ web/frontend/src/types/
git commit -m "feat: add Zustand store, API helpers (mapApi, chatApi)"
```

---

## Task 9: MapView Component

**Files:**
- Create: `web/frontend/src/components/MapView/MapView.tsx`

- [ ] **Step 1: Create `MapView.tsx`**

```tsx
import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useMapStore } from '../../store/useMapStore'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN as string

// Viridis-inspired color stops (low → high)
const COLOR_STOPS = [
  [0,   '#1a237e'],
  [0.2, '#1565c0'],
  [0.4, '#00838f'],
  [0.6, '#2e7d32'],
  [0.8, '#f9a825'],
  [1.0, '#e65100'],
]

export function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<mapboxgl.Map | null>(null)
  const { geojsonData, stats, selectedVariable } = useMapStore()

  // Initialize map once
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return
    mapRef.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-96, 38],
      zoom: 3.5,
    })
    mapRef.current.addControl(new mapboxgl.NavigationControl(), 'top-right')
  }, [])

  // Update choropleth when data changes
  useEffect(() => {
    const map = mapRef.current
    if (!map || !geojsonData) return

    const onLoad = () => {
      // Remove existing layers
      if (map.getLayer('choropleth-fill')) map.removeLayer('choropleth-fill')
      if (map.getLayer('choropleth-line')) map.removeLayer('choropleth-line')
      if (map.getSource('acs-data')) map.removeSource('acs-data')

      map.addSource('acs-data', { type: 'geojson', data: geojsonData })

      const minVal = stats?.min ?? 0
      const maxVal = stats?.max ?? 1
      const range = maxVal - minVal || 1

      // Build Mapbox interpolation expression
      const stops: (number | string)[] = []
      COLOR_STOPS.forEach(([ratio, color]) => {
        stops.push(minVal + (ratio as number) * range, color as string)
      })

      map.addLayer({
        id: 'choropleth-fill',
        type: 'fill',
        source: 'acs-data',
        paint: {
          'fill-color': [
            'interpolate', ['linear'],
            ['coalesce', ['get', selectedVariable], minVal],
            ...stops,
          ],
          'fill-opacity': 0.75,
        },
      })

      map.addLayer({
        id: 'choropleth-line',
        type: 'line',
        source: 'acs-data',
        paint: { 'line-color': '#2a2d3e', 'line-width': 0.5 },
      })

      // Tooltip on hover
      const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false })
      map.on('mousemove', 'choropleth-fill', (e) => {
        map.getCanvas().style.cursor = 'pointer'
        const props = e.features?.[0]?.properties ?? {}
        const name = props.NAME ?? props.STUSPS ?? props.ZCTA5CE20 ?? ''
        const val = props[selectedVariable]
        popup.setLngLat(e.lngLat)
          .setHTML(`<div class="text-sm"><b>${name}</b><br/>${selectedVariable}: ${val ?? 'N/A'}</div>`)
          .addTo(map)
      })
      map.on('mouseleave', 'choropleth-fill', () => {
        map.getCanvas().style.cursor = ''
        popup.remove()
      })
    }

    if (map.isStyleLoaded()) onLoad()
    else map.once('load', onLoad)
  }, [geojsonData, stats, selectedVariable])

  return <div ref={mapContainer} className="absolute inset-0" />
}
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/components/MapView/
git commit -m "feat: add MapView component with Mapbox choropleth + tooltip"
```

---

## Task 10: ConfigPanel Component

**Files:**
- Create: `web/frontend/src/components/ConfigPanel/ConfigPanel.tsx`

- [ ] **Step 1: Create `ConfigPanel.tsx`**

```tsx
import { useEffect } from 'react'
import { useMapStore } from '../../store/useMapStore'
import { fetchVariables, fetchRegions, fetchMapData } from '../../api/mapApi'
import type { Level } from '../../types'

const YEARS = [2019, 2020, 2021, 2022, 2023]
const LEVELS: { value: Level; label: string }[] = [
  { value: 'state',   label: 'State' },
  { value: 'county',  label: 'County' },
  { value: 'zipcode', label: 'Zipcode' },
]

const SELECT_CLS = 'bg-panel border border-border text-text rounded px-2 py-1 text-sm focus:outline-none focus:border-primary'
const BTN_CLS = 'w-full mt-3 py-2 rounded-lg font-semibold text-white text-sm ' +
  'bg-gradient-to-r from-primary to-accent hover:opacity-90 transition'

export function ConfigPanel() {
  const {
    level, setLevel,
    selectedState, setSelectedState,
    selectedCounty, setSelectedCounty,
    selectedVariable, setSelectedVariable,
    selectedYear, setSelectedYear,
    availableVariables, setAvailableVariables,
    availableCounties, setAvailableCounties,
    setMapData,
  } = useMapStore()

  // Load variable list once
  useEffect(() => {
    fetchVariables().then(setAvailableVariables)
  }, [])

  // Load counties when state changes
  useEffect(() => {
    if (selectedState) {
      fetchRegions('county', selectedState).then(setAvailableCounties)
    } else {
      setAvailableCounties([])
      setSelectedCounty(null)
    }
  }, [selectedState])

  const US_STATES = [
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC',
  ]

  async function handleUpdate() {
    try {
      const result = await fetchMapData({
        level,
        variable: selectedVariable,
        year: selectedYear,
        state: selectedState ?? undefined,
        county: selectedCounty ?? undefined,
      })
      setMapData(result.geojson, result.stats)
    } catch (e) {
      alert(`Error: ${e}`)
    }
  }

  return (
    <div className="absolute top-0 left-0 right-0 z-10 m-3">
      <div className="bg-panel/90 backdrop-blur border border-border rounded-xl p-4 shadow-xl">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {/* Year */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">Year</label>
            <select className={SELECT_CLS} value={selectedYear}
              onChange={e => setSelectedYear(Number(e.target.value))}>
              {YEARS.map(y => <option key={y}>{y}</option>)}
            </select>
          </div>
          {/* Level */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">Level</label>
            <select className={SELECT_CLS} value={level}
              onChange={e => setLevel(e.target.value as Level)}>
              {LEVELS.map(l => <option key={l.value} value={l.value}>{l.label}</option>)}
            </select>
          </div>
          {/* Variable */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">Variable</label>
            <select className={SELECT_CLS} value={selectedVariable}
              onChange={e => setSelectedVariable(e.target.value)}>
              {availableVariables.map(v => <option key={v}>{v}</option>)}
            </select>
          </div>
          {/* State */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">State</label>
            <select className={SELECT_CLS} value={selectedState ?? ''}
              onChange={e => setSelectedState(e.target.value || null)}>
              <option value="">All States</option>
              {US_STATES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
          {/* County (only when level=county or zipcode) */}
          {level !== 'state' && (
            <div className="flex flex-col gap-1">
              <label className="text-muted text-xs">County</label>
              <select className={SELECT_CLS} value={selectedCounty ?? ''}
                onChange={e => setSelectedCounty(e.target.value || null)}>
                <option value="">All Counties</option>
                {availableCounties.map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
        </div>
        <button className={BTN_CLS} onClick={handleUpdate}>
          Update Map &amp; Stats
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/components/ConfigPanel/
git commit -m "feat: add ConfigPanel with level/variable/state/county selectors"
```

---

## Task 11: StatsCard Component

**Files:**
- Create: `web/frontend/src/components/StatsCard/StatsCard.tsx`

- [ ] **Step 1: Create `StatsCard.tsx`**

```tsx
import { useMapStore } from '../../store/useMapStore'

function fmt(n: number) {
  return n >= 1000 ? n.toLocaleString('en-US', { maximumFractionDigits: 0 }) : n.toFixed(2)
}

export function StatsCard() {
  const stats = useMapStore(s => s.stats)
  if (!stats) return null

  const rows = [
    { label: 'Min',    value: fmt(stats.min) },
    { label: 'Max',    value: fmt(stats.max) },
    { label: 'Mean',   value: fmt(stats.mean) },
    { label: 'Median', value: fmt(stats.median) },
  ]

  return (
    <div className="absolute bottom-8 left-4 z-10 w-48
      bg-panel/80 backdrop-blur border border-border rounded-xl p-3 shadow-xl">
      <p className="text-xs text-muted mb-2 truncate">{stats.variable}</p>
      {rows.map(r => (
        <div key={r.label} className="flex justify-between text-xs mb-1">
          <span className="text-muted">{r.label}</span>
          <span className="text-text font-mono">{r.value}</span>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/components/StatsCard/
git commit -m "feat: add StatsCard floating bottom-left overlay"
```

---

## Task 12: ChatPanel Component

**Files:**
- Create: `web/frontend/src/components/ChatPanel/ChatPanel.tsx`

- [ ] **Step 1: Create `ChatPanel.tsx`**

```tsx
import { useRef, useState } from 'react'
import { useMapStore } from '../../store/useMapStore'
import { streamChat } from '../../api/chatApi'
import type { ChatDataPayload } from '../../types'

export function ChatPanel() {
  const { chatHistory, chatOpen, addChatMessage, setChatOpen, updateFromChatData } = useMapStore()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [aiDraft, setAiDraft] = useState('')
  // useRef to avoid stale closure in onDone callback — always holds latest draft value
  const aiDraftRef = useRef('')
  const stopRef = useRef<(() => void) | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  function send() {
    if (!input.trim() || loading) return
    const userMsg = { role: 'user' as const, content: input.trim() }
    addChatMessage(userMsg)
    setInput('')
    setLoading(true)
    aiDraftRef.current = ''
    setAiDraft('')

    stopRef.current = streamChat(
      userMsg.content,
      chatHistory,
      {
        onText(text) {
          aiDraftRef.current = text   // keep ref in sync
          setAiDraft(text)
        },
        onData(payload: ChatDataPayload) { updateFromChatData(payload) },
        onError(msg) {
          aiDraftRef.current = `Error: ${msg}`
          setAiDraft(`Error: ${msg}`)
        },
        onDone() {
          setLoading(false)
          // Read from ref — not from closed-over state — to get the final value
          if (aiDraftRef.current) {
            addChatMessage({ role: 'assistant', content: aiDraftRef.current })
          }
          aiDraftRef.current = ''
          setAiDraft('')
          setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
        },
      }
    )
  }

  // Minimized bubble
  if (!chatOpen) {
    return (
      <button
        onClick={() => setChatOpen(true)}
        className="absolute bottom-8 right-4 z-20 w-14 h-14 rounded-full
          bg-gradient-to-br from-primary to-accent shadow-xl flex items-center justify-center
          text-white text-2xl hover:scale-105 transition"
        title="Open Chat Agent"
      >
        💬
      </button>
    )
  }

  return (
    <div className="absolute bottom-8 right-4 z-20 w-80 md:w-96
      bg-panel/90 backdrop-blur border border-border rounded-2xl shadow-2xl
      flex flex-col overflow-hidden" style={{ height: '420px' }}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <span className="text-sm font-semibold text-text">Chat Agent</span>
        <div className="flex gap-2">
          <button onClick={() => setChatOpen(false)}
            className="text-muted hover:text-text text-lg leading-none">─</button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2 text-sm">
        {chatHistory.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-xl px-3 py-2 ${
              m.role === 'user'
                ? 'bg-primary text-white'
                : 'bg-border text-text'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-xl px-3 py-2 bg-border text-text whitespace-pre-wrap">
              {aiDraft || <span className="animate-pulse text-muted">思考中...</span>}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border p-2 flex gap-2">
        <input
          className="flex-1 bg-bg border border-border rounded-lg px-3 py-1.5 text-sm text-text
            placeholder:text-muted focus:outline-none focus:border-primary"
          placeholder="问我任何 ACS 数据问题..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          disabled={loading}
        />
        <button
          onClick={send}
          disabled={loading}
          className="px-3 py-1.5 rounded-lg text-sm font-medium text-white
            bg-gradient-to-r from-primary to-accent hover:opacity-90 disabled:opacity-50"
        >
          发送
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/components/ChatPanel/
git commit -m "feat: add ChatPanel floating SSE chat window"
```

---

## Task 13: DataTable Component

**Files:**
- Create: `web/frontend/src/components/DataTable/DataTable.tsx`

- [ ] **Step 1: Create `DataTable.tsx`**

```tsx
import { AgGridReact } from 'ag-grid-react'
import 'ag-grid-community/styles/ag-grid.css'
import 'ag-grid-community/styles/ag-theme-balham.css'
import { useMapStore } from '../../store/useMapStore'

export function DataTable() {
  const geojsonData = useMapStore(s => s.geojsonData)

  if (!geojsonData) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted text-sm">
        先在 Map Visualization 标签中查询数据
      </div>
    )
  }

  // Extract row data from GeoJSON features
  const rows = geojsonData.features.map(f => f.properties ?? {})
  const allKeys = rows.length > 0 ? Object.keys(rows[0]) : []
  const colDefs = allKeys.map(k => ({
    field: k,
    sortable: true,
    filter: true,
    resizable: true,
  }))

  return (
    <div className="flex-1 ag-theme-balham-dark" style={{ height: '100%' }}>
      <AgGridReact rowData={rows} columnDefs={colDefs} pagination paginationPageSize={50} />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/components/DataTable/
git commit -m "feat: add DataTable AG Grid tab view"
```

---

## Task 14: App Assembly — Wire All Components Together

**Files:**
- Modify: `web/frontend/src/App.tsx`

- [ ] **Step 1: Replace `App.tsx` with full layout**

```tsx
import { useState } from 'react'
import { MapView }     from './components/MapView/MapView'
import { ConfigPanel } from './components/ConfigPanel/ConfigPanel'
import { ChatPanel }   from './components/ChatPanel/ChatPanel'
import { StatsCard }   from './components/StatsCard/StatsCard'
import { DataTable }   from './components/DataTable/DataTable'

type Tab = 'map' | 'table'

export default function App() {
  const [tab, setTab] = useState<Tab>('map')

  const tabCls = (t: Tab) =>
    `px-4 py-2 text-sm font-medium transition border-b-2 ${
      tab === t
        ? 'border-primary text-primary'
        : 'border-transparent text-muted hover:text-text'
    }`

  return (
    <div className="bg-bg min-h-screen flex flex-col text-text">
      {/* Tab bar */}
      <nav className="flex border-b border-border px-4 pt-2 bg-panel z-30 relative">
        <button className={tabCls('map')}   onClick={() => setTab('map')}>
          Map Visualization
        </button>
        <button className={tabCls('table')} onClick={() => setTab('table')}>
          Data Table
        </button>
      </nav>

      {/* Map tab — full screen relative container */}
      {tab === 'map' && (
        <div className="flex-1 relative overflow-hidden">
          <MapView />
          <ConfigPanel />
          <StatsCard />
          <ChatPanel />
        </div>
      )}

      {/* Data table tab */}
      {tab === 'table' && (
        <div className="flex-1 flex flex-col p-4">
          <DataTable />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Add Mapbox GL CSS to `src/index.css`**

Ensure `index.css` (generated by Vite) includes:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 3: Run and verify the full app**

Start both servers:

```bash
# Terminal 1 — backend
cd I:/LLM_proj/llm_subproj/Dash_Agent/web/backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd I:/LLM_proj/llm_subproj/Dash_Agent/web/frontend
npm run dev
```

Open `http://localhost:5173/`:
- [ ] Map renders with Mapbox dark-v11 base style
- [ ] ConfigPanel shows Year/Level/Variable/State dropdowns
- [ ] Clicking "Update Map & Stats" fetches data and colors the map
- [ ] StatsCard appears bottom-left with Min/Max/Mean/Median
- [ ] Chat bubble visible bottom-right; expands on click
- [ ] Typing a question and submitting triggers SSE; map updates from `data` event
- [ ] Data Table tab shows AG Grid table from last query

- [ ] **Step 4: Final commit**

```bash
cd I:/LLM_proj/llm_subproj/Dash_Agent
git add web/frontend/src/App.tsx web/frontend/src/index.css
git commit -m "feat: assemble full ACS Data Explorer app with map, chat, and data table"
```

---

## Running the App (Quick Reference)

```bash
# 1. Activate venv
cd I:/LLM_proj/llm_subproj/Dash_Agent
# (activate your venv)

# 2. Start backend
cd web/backend
uvicorn main:app --reload --port 8000

# 3. Start frontend (new terminal)
cd web/frontend
npm run dev

# 4. Open browser
# http://localhost:5173
```
