from __future__ import annotations

from typing import Any, Dict, List, Optional

import psycopg2.extras

from db_utils import db_conn

TABLE = "public.acs_data_all"

# 位置过滤字段白名单（防止列名注入）
_LOCATION_FIELDS = {"state", "county", "city", "zipcode"}


# ── 1. 查询所有字段名 ────────────────────────────────────────────────────────

def get_column_names() -> List[str]:
    """返回 acs_data_all 表的所有字段名（按原始顺序）。"""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'acs_data_all'
                ORDER BY ordinal_position;
            """)
            return [row[0] for row in cur.fetchall()]


# ── 2. 按位置 + 时间查询指定变量 ─────────────────────────────────────────────

def query_acs_data(
    variables: Optional[List[str]] = None,
    *,
    zipcode: Optional[str] = None,
    city: Optional[str] = None,
    county: Optional[str] = None,
    state: Optional[str] = None,
    year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """查询 acs_data_all 中指定位置、时间的变量值。

    Parameters
    ----------
    variables:
        要返回的列名列表。None 表示返回所有列。
    zipcode / city / county / state:
        位置过滤条件，可组合使用，均为精确匹配（大小写不敏感）。
    year:
        年份过滤，None 表示不限年份。

    Returns
    -------
    List[dict]  每行数据为一个字典 {列名: 值}。
    """
    all_columns = get_column_names()

    # 校验并构建 SELECT 列表
    if variables is None:
        select_clause = "*"
    else:
        invalid = [v for v in variables if v not in all_columns]
        if invalid:
            raise ValueError(f"未知字段名: {invalid}。可用字段请调用 get_column_names()。")
        # 位置/时间标识列始终带上，方便调用方识别行
        anchor = [c for c in ("zipcode", "city", "county", "state", "year") if c not in variables]
        cols = anchor + variables
        select_clause = ", ".join(cols)

    # 构建 WHERE 条件（使用参数化查询防注入）
    conditions: List[str] = []
    params: List[Any] = []

    loc_filters = {"zipcode": zipcode, "city": city, "county": county, "state": state}
    for col, val in loc_filters.items():
        if val is not None:
            conditions.append(f"LOWER({col}) = LOWER(%s)")
            params.append(val)

    if year is not None:
        conditions.append("year = %s")
        params.append(year)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"SELECT {select_clause} FROM {TABLE} {where_clause};"

    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    # 查询所有字段名
    cols = get_column_names()
    print(f"共 {len(cols)} 个字段：", cols[:6], "…")

    # 查询 zipcode=10001，2020 年的教育和收入变量
    rows = query(
        variables=["pct_bachelor", "pct_high_school_or_higher", "median_income", "population"],
        # zipcode="10001",
        city="New York",
        year=2020,
    )
    for i, r in enumerate(rows):
        print(i, r)
        if i > 10:
            break
