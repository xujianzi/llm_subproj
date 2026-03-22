"""
db_tools.py
将 query_db 中的函数封装为 LLM tool schema（Pydantic BaseModel）。

用法：
    from extract_agent import AgentforExtraction, build_client, MODEL_NAME
    from db_tools import GetColumnNames, QueryACSData, execute

    agent = AgentforExtraction(MODEL_NAME, build_client())

    # LLM 决定调用哪些工具，返回模型实例列表
    results = agent.call_multi(user_prompt, [GetColumnNames, QueryACSData], system_prompt)

    # 将每个实例路由到对应的 query_db 函数
    for tool_result in results:
        data = execute(tool_result)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import query_db


# ── Tool 1: 查询所有字段名 ────────────────────────────────────────────────────

class GetColumnNames(BaseModel):
    """查询 acs_data_all 表的所有可用字段名。在不确定有哪些变量时优先调用此工具。"""
    # 无参数；properties 为空对象，配合 call_multi 使用（call_multi 对 required 做了兜底）
    pass


# ── Tool 2: 按位置 + 时间查询变量 ─────────────────────────────────────────────

class QueryACSData(BaseModel):
    """按位置（zipcode / city / county / state）和年份查询 ACS 人口统计指标。"""

    variables: Optional[List[str]] = Field(
        default=None,
        description=(
            "要查询的字段名列表，例如 ['pct_bachelor', 'median_income', 'population']。"
            "字段名须与 GetColumnNames 返回的名称完全一致。"
            "不填则返回全部列。"
        ),
    )
    zipcode: Optional[str] = Field(
        default=None,
        description="邮政编码，如 '10001'。与 city/county/state 可组合使用。",
    )
    city: Optional[str] = Field(
        default=None,
        description="城市名，如 'New York'，大小写不敏感。",
    )
    county: Optional[str] = Field(
        default=None,
        description="县名，如 'NEW YORK'，大小写不敏感。",
    )
    state: Optional[str] = Field(
        default=None,
        description="州缩写，如 'NY'，大小写不敏感。",
    )
    year: Optional[int] = Field(
        default=None,
        description="年份，如 2020。不填则返回该位置所有年份的数据。",
    )


# ── 执行路由 ──────────────────────────────────────────────────────────────────

def execute(tool_result: BaseModel) -> Any:
    """将 LLM 返回的 tool 实例路由到对应的 query_db 函数并执行。

    Parameters
    ----------
    tool_result : GetColumnNames | QueryACSData
        agent.call_multi() 解析出的模型实例。

    Returns
    -------
    List[str] 或 List[dict]，取决于调用的工具。
    """
    if isinstance(tool_result, GetColumnNames):
        return query_db.get_column_names()

    if isinstance(tool_result, QueryACSData):
        return query_db.query(
            variables=tool_result.variables,
            zipcode=tool_result.zipcode,
            city=tool_result.city,
            county=tool_result.county,
            state=tool_result.state,
            year=tool_result.year,
        )

    raise TypeError(f"未知的 tool 类型: {type(tool_result)}")


if __name__ == "__main__":
    from extract_agent import AgentforExtraction, build_client, MODEL_NAME

    SYSTEM = """你是一个美国社区调查（ACS）数据库查询助手。数据库语言为英文，用户提问为中文，你需要完成语言映射后再调用工具。

## 工具使用规则
1. 若不确定有哪些可用字段，先调用 GetColumnNames 获取完整字段列表。
2. 确认字段后，调用 QueryACSData 查询数据，所有参数必须使用英文。

## 语言映射规则
**地名**：将中文地名翻译为数据库中的英文值。
- 城市：纽约 → New York，洛杉矶 → Los Angeles，芝加哥 → Chicago
- 州：纽约州 → NY，加州 → CA，德州 → TX
- 可直接使用用户提供的邮政编码（zipcode）

**字段名**：根据用户描述的中文含义，从 GetColumnNames 返回的字段列表中匹配最接近的英文字段名。
- 常见对应：本科学历 → pct_bachelor，中位收入 → median_income，人口 → population
- 贫困率 → pct_below_poverty，失业率 → pct_unemployed，白人比例 → pct_white

## 注意
- variables 字段名必须与数据库字段完全一致，不能自行构造。
- 位置参数（city/state/county/zipcode）使用英文，大小写不敏感。
"""

    agent = AgentforExtraction(MODEL_NAME, build_client())
    user_q = "帮我查一下纽约市 2020 年的本科学历比例和中位收入"

    answer = agent.run(user_q, [GetColumnNames, QueryACSData], execute, SYSTEM)
    print("\n[最终回复]", answer)
