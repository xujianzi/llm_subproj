"""
下载 Census 500K 简化版边界文件并转换为 GeoJSON。
使用 pyshp 读取 shapefile，无需 ogr2ogr。
"""
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
import shapefile  # pip install pyshp

BASE = Path(__file__).parent.parent
GEODATA = BASE / "geodata"
DATA = Path(__file__).parent
GEODATA.mkdir(exist_ok=True)

CENSUS_BASE = "https://www2.census.gov/geo/tiger/GENZ2022/shp"

DOWNLOADS = {
    "states":   f"{CENSUS_BASE}/cb_2022_us_state_20m.zip",
    "counties": f"{CENSUS_BASE}/cb_2022_us_county_20m.zip",
    "zcta":     "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip",
}


def shp_zip_to_geojson(url: str, keep_fields: list) -> dict:
    """下载 shapefile zip，转换为 GeoJSON FeatureCollection。"""
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # 找到 .shp .dbf .shx 文件（同前缀）
        shp_name = next(n for n in zf.namelist() if n.endswith(".shp"))
        prefix = shp_name[:-4]
        shp_data = zf.read(shp_name)
        dbf_data = zf.read(prefix + ".dbf")
        shx_data = zf.read(prefix + ".shx")

    sf = shapefile.Reader(
        shp=io.BytesIO(shp_data),
        dbf=io.BytesIO(dbf_data),
        shx=io.BytesIO(shx_data),
    )

    field_names = [f[0] for f in sf.fields[1:]]  # 跳过 DeletionFlag
    features = []
    for sr in sf.shapeRecords():
        props = dict(zip(field_names, sr.record))
        # 只保留需要的字段
        props = {k: v for k, v in props.items() if k in keep_fields}
        geom = sr.shape.__geo_interface__
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": geom,
        })

    print(f"  -> {len(features)} features")
    return {"type": "FeatureCollection", "features": features}


def main():
    # ── States ────────────────────────────────────────────────────────────────
    states_gj = shp_zip_to_geojson(DOWNLOADS["states"], ["STUSPS", "NAME"])
    out = GEODATA / "states.geojson"
    out.write_text(json.dumps(states_gj), encoding="utf-8")
    print(f"Wrote {out}")

    # ── Counties ──────────────────────────────────────────────────────────────
    counties_gj = shp_zip_to_geojson(DOWNLOADS["counties"], ["STUSPS", "NAME", "STATEFP"])
    out = GEODATA / "counties.geojson"
    out.write_text(json.dumps(counties_gj), encoding="utf-8")
    print(f"Wrote {out}")

    # ── ZCTA ──────────────────────────────────────────────────────────────────
    zcta_gj = shp_zip_to_geojson(DOWNLOADS["zcta"], ["ZCTA5CE20"])
    out = GEODATA / "zcta_all.geojson"
    out.write_text(json.dumps(zcta_gj), encoding="utf-8")
    print(f"Wrote {out}")

    # ── county_names.json ─────────────────────────────────────────────────────
    county_names = defaultdict(list)
    fips_to_abbr = {}
    for feat in counties_gj["features"]:
        p = feat["properties"]
        state_abbr = p.get("STUSPS", "")
        county_name = p.get("NAME", "")
        statefp = p.get("STATEFP", "")
        if state_abbr and county_name:
            county_names[state_abbr].append(county_name)
        if statefp and state_abbr:
            fips_to_abbr[statefp] = state_abbr

    for s in county_names:
        county_names[s] = sorted(county_names[s])

    out = DATA / "county_names.json"
    out.write_text(json.dumps(dict(county_names), indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    # ── state_fips.json ───────────────────────────────────────────────────────
    state_fips = {abbr: fips for fips, abbr in fips_to_abbr.items()}
    out = DATA / "state_fips.json"
    out.write_text(json.dumps(state_fips, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    print("\nAll geodata files generated.")


if __name__ == "__main__":
    main()
