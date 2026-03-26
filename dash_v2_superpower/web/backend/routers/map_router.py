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
    level: str = Query(..., pattern="^(state|county|zipcode)$"),
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
