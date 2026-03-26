#!/usr/bin/env python3
"""
HarnessAgent — OpenAI Chat Completions API 版本

架构概览
────────────────────────────────────────────────────────────────────
  用户输入
    ↓
  input_messages (上下文列表，原地扩展)
    ↓
  client.chat.completions.create(model, messages=[system]+input, tools)
    ↓
  choice.message → 追加到 input_messages（维持上下文连贯）
    ↓
  检查 message.tool_calls → dispatch(工具名) → tool 消息追加
    ↓ (无 tool_calls)
  返回 message.content

功能模块
────────────────────────────────────────────────────────────────────
  1. Chat Completions API  — client.chat.completions.create
  2. 上下文压缩            — micro_compact + auto_compact 两层压缩
  3. 工具派发 (dispatch)   — TOOL_HANDLERS 字典统一管理 工具名 → 函数 映射

【Chat Completions API 核心概念速查】
  调用:   client.chat.completions.create(model, messages, tools)
  系统提示: {"role":"system","content":...} 作为 messages 第一条
  工具格式: {"type":"function","function":{"name":...,"parameters":...}}  ← 嵌套
  工具调用检测: choice.message.tool_calls
  工具调用字段: tc.function.name / tc.function.arguments(JSON串) / tc.id
  工具结果格式: {"role":"tool","tool_call_id":...,"content":...}
  文本读取: choice.message.content
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion 

from pydantic import BaseModel, Field
from typing import List, Literal, Type

from Settings import get_settings
from model_registry import resolve_model
import Settings

from db_models import GetColumnNames, QueryACSData
from query_db import get_column_names, query_acs_data

_cfg = get_settings()

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL = resolve_model("qwen-plus")
BASE_URL   = _cfg.dashscope_base_url
API_KEY    = _cfg.openai_api_key
WORKDIR    = Path(__file__).parent

# 上下文压缩阈值（估算 token 数超过此值触发 auto_compact）
COMPACT_THRESHOLD = 50_000
# micro_compact 保留最近几条工具输出不压缩
KEEP_RECENT       = 3
# 对话记录存储目录
TRANSCRIPT_DIR    = WORKDIR / ".transcripts"

# 检测操作系统，决定 shell 类型（影响 run_bash 和系统提示）
_IS_WINDOWS = sys.platform == "win32"
_SHELL_NAME = "PowerShell" if _IS_WINDOWS else "bash"

# ── 系统提示 ──────────────────────────────────────────────────────────────────
# Responses API 的系统指令通过独立的 instructions 参数传入，不混入 input 列表
SYSTEM = (
    f"你是一个美国社区调查（ACS）数据库查询助手。数据库语言为英文，用户提问为中文，你需要完成语言映射后再调用工具。"
    f"工作目录为 {WORKDIR}。 "
    "Use tools to solve tasks if needed. " # Act, don't explain.
)

def build_client() -> OpenAI:
    """
    Build OpenAI client.
    """
    api_key = API_KEY
    
    if not api_key:
        sys.exit(
            "ERROR: No API key found.\n"
            "Set DASHSCOPE_API_KEY (or OPENAI_API_KEY) in your environment or .env file."
        )
    kwargs = {"api_key":api_key}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)

client = build_client()

def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数（约 4 字符 / token），用于触发 auto_compact。"""
    return len(json.dumps(messages, default=str)) // 4

def micro_compact(messages: list) -> None:
    """
    Layer 1 — 微压缩：将历史工具结果中较长的输出截断为摘要占位符。

    原理：保留最新 KEEP_RECENT 条工具输出不变，
         将更早的输出内容替换为 "[Previous: used <tool_name>]"，
         大幅减少旧工具结果占用的 token，同时保留调用痕迹供模型参考。

    Chat Completions 上下文结构：
      input_messages 是列表，包含：
        - {"role":"system",    "content":...}                        ← 系统提示（不在此列表中）
        - {"role":"user",      "content":...}                        ← 用户消息
        - {"role":"assistant", "content":..., "tool_calls":[...]}    ← 模型回复（含工具调用）
        - {"role":"tool",      "tool_call_id":..., "content":...}    ← 工具结果
    """
    # 1. 收集所有工具结果消息的引用（role=="tool"）
    tool_msgs = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]

    if len(tool_msgs) <= KEEP_RECENT:
        return  # 数量未超出，无需压缩

    # 2. 构建 tool_call_id → function name 的映射（从 assistant 消息的 tool_calls 中提取）
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    id_to_name[tc["id"]] = tc["function"]["name"]

    # 3. 对超出保留窗口的旧输出进行截断（原地修改 dict）
    to_compress = tool_msgs[:-KEEP_RECENT]
    for msg in to_compress:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 100:
            name = id_to_name.get(msg.get("tool_call_id"), "unknown")
            msg["content"] = f"[Previous: used {name}]"

def auto_compact(messages: list) -> list:
    """
    Layer 2 — 自动压缩：当 token 估算超过阈值时，
               将整段对话摘要化，替换为 2 条简洁消息，同时把原始记录落盘。

    流程：
      1. 将当前 input_messages 序列化并写入 .transcripts/ 目录（防丢失）
      2. 调用 LLM（client.responses.create）对对话做摘要
      3. 用摘要替换整个上下文，大幅缩减 token 占用
    """
    # 步骤 1：落盘原始对话
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for item in messages:
            f.write(json.dumps(item, default=str) + "\n")
    print(f"[transcript saved → {transcript_path}]")

    # 步骤 2：调用 LLM 生成摘要
    conversation_text = json.dumps(messages, default=str)[:80_000]
    summary_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a conversation summarizer. Be concise and factual."},
            {"role": "user", "content": "Summarize this agent conversation for continuity.\n\n" + conversation_text},
        ],
        max_tokens=2000,
    )
    summary = summary_response.choices[0].message.content

    # 步骤 3：用 2 条消息替换整段上下文
    return [
        {
            "role": "user",
            "content": (
                f"[Context compressed. Full transcript: {transcript_path}]\n\n"
                f"{summary}"
            ),
        },
        {
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing.",
        },
    ]

# ══════════════════════════════════════════════════════════════════════════════
# 工具派发 (Tool Dispatch)
# ══════════════════════════════════════════════════════════════════════════════
# TOOL_HANDLERS 是工具名称 → 实现函数的中央映射表（dispatch table）。
# agent_loop 通过查询此表完成 工具名 → 函数调用 的派发，无需 if/elif 链。
# 新增工具：在此添加一行 + 在 TOOLS 列表中添加定义，不需要改 agent_loop。

TOOL_HANDLERS: dict[str, callable] = {
    "QueryACSData":     lambda **kw: query_acs_data(**kw),
    "GetColumnNames":   lambda **kw: get_column_names(**kw),
    "Compact":          lambda **kw: "__COMPACT__",   # 特殊标记，在 agent_loop 中处理 Manual compression requested.
}

# ══════════════════════════════════════════════════════════════════════════════
# 工具定义 (Tool Definitions for Chat Completions API)
# ══════════════════════════════════════════════════════════════════════════════
# 【Chat Completions 工具格式】嵌套结构：
#   {"type":"function", "function":{"name":..., "description":..., "parameters":...}}

TOOL_MODELS = [GetColumnNames, QueryACSData]

def generate_tools(model_cls: Type[BaseModel]) -> dict:
    schema = model_cls.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": schema["title"],
            "description": schema.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        },
    }

TOOLS = [generate_tools(m) for m in TOOL_MODELS]

acompact_schema = {
    "type": "function",
    "function": {
        "name": "Compact",
        "description": "Trigger manual conversation compression.",
        "parameters": {
            "type": "object",
            "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}},
            "required": [],
        },
    },
}

TOOLS.append(acompact_schema)

# ══════════════════════════════════════════════════════════════════════════════
# Agent 主循环 (Agent Loop)
# ══════════════════════════════════════════════════════════════════════════════

def agent_loop(input_messages: list, model: str = None) -> None:
    """
    Chat Completions API 的核心 Agentic Loop。

    【循环逻辑】
    ┌──────────────────────────────────────────────────────────────┐
    │  while True:                                                 │
    │    0. 上下文压缩检查（micro_compact + 超阈值则 auto_compact）│
    │    1. chat.completions.create([system]+input_messages)      │
    │    2. input_messages += assistant message  (维持上下文)      │
    │    3. 检查 message.tool_calls                               │
    │       - 无 → 结束循环                                        │
    │       - 有 → dispatch 执行，追加 tool 消息，继续             │
    └──────────────────────────────────────────────────────────────┘

    参数:
        input_messages: 上下文列表，每轮循环后原地扩展（不含 system 消息）。
        model:          模型名称，默认使用全局 MODEL。
    返回:
        无。结果通过 input_messages 原地传递，调用方自行读取最后一条。
    """
    _model = model or MODEL
    system_msg = {"role": "system", "content": SYSTEM}

    while True:
        # ── 步骤 0：上下文压缩 ──────────────────────────────────────────────
        micro_compact(input_messages)
        if estimate_tokens(input_messages) > COMPACT_THRESHOLD:
            print("[auto_compact triggered]")
            input_messages[:] = auto_compact(input_messages)

        # ── 步骤 1：调用 Chat Completions API ───────────────────────────────
        response = client.chat.completions.create(
            model=_model,
            messages=[system_msg] + input_messages,
            tools=TOOLS,
        )

        # ── 步骤 2：将 assistant 消息追加到上下文 ───────────────────────────
        msg = response.choices[0].message
        input_messages.append(msg.model_dump(exclude_unset=False))

        # ── 步骤 3：检查是否有工具调用 ──────────────────────────────────────
        if not msg.tool_calls:
            return
        manual_compact = False
        # ── 步骤 4：通过 dispatch 执行工具，逐条追加 tool 消息 ──────────────
        for tc in msg.tool_calls:
            if tc.function.name == "Compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(tc.function.name)
                try:
                    output = handler(**json.loads(tc.function.arguments)) if handler else f"Unknown tool: {tc.function.name}"
                except Exception as e:
                    output = f"Error: {e}"

            print(f"> {tc.function.name}: {str(output)[:200]}")
            input_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(output),
            })
            if manual_compact:
                print("Manual compact triggered.")
                input_messages[:] = auto_compact(input_messages)


if __name__ == "__main__":
    ## smoke test 
    user_query = "数据库中有哪些表"
    history = [{"role": "user", "content": user_query}]
    agent_loop(history)
    print(history[-1]["content"])
    # print(history)
    user_query2 = "好"
    history.append({"role": "user", "content": user_query2})
    agent_loop(history)
    print(history[-1]["content"])
