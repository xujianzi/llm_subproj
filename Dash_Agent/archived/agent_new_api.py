#!/usr/bin/env python3
"""
HarnessAgent — OpenAI Responses API 版本

架构概览
────────────────────────────────────────────────────────────────────
  用户输入
    ↓
  input_messages (上下文列表，原地扩展)
    ↓
  client.responses.create(instructions, input, tools)
    ↓
  response.output → 追加到 input_messages（维持上下文连贯）
    ↓
  遍历 output → type=="function_call" → dispatch(工具名) → function_call_output
    ↓ (无 function_call)
  返回 output_text

功能模块
────────────────────────────────────────────────────────────────────
  1. OpenAI Responses API  — client.responses.create (新规范)
  2. 上下文压缩            — micro_compact + auto_compact 两层压缩
  3. Skill 调用            — call_skill 工具，按名称执行预定义技能
  4. 子 Agent              — spawn_agent 工具，创建独立子任务 agent 实例
  5. 工具派发 (dispatch)   — TOOL_HANDLERS 字典统一管理 工具名 → 函数 映射

【Responses API 核心概念速查】
  调用:   client.responses.create(model, instructions, input, tools)
  上下文: input_messages += response.output   ← 关键：把模型输出追加回去
  工具格式: {"type":"function", "name":..., "parameters":...}  ← 扁平，无嵌套
  工具调用检测: item.type == "function_call"
  工具调用字段: item.name / item.arguments(JSON串) / item.call_id
  工具结果格式: {"type":"function_call_output", "call_id":..., "output":...}
  文本快捷读取: response.output_text
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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
    Layer 1 — 微压缩：将历史 function_call_output 中较长的输出截断为摘要占位符。

    原理：保留最新 KEEP_RECENT 条工具输出不变，
         将更早的输出内容替换为 "[Previous: used <tool_name>]"，
         大幅减少旧工具结果占用的 token，同时保留调用痕迹供模型参考。

    Responses API 上下文结构：
      input_messages 是混合列表，包含：
        - dict {"role":"user", "content":...}               ← 用户消息
        - tool_call 对象 (type="tool_call") ← 模型工具调用
        - dict {"type":"tool_result", "call_id":..., "output":...} ← 工具结果
        - ResponseOutputMessage 对象 (type="message")       ← 模型文本回复
    """
    # Collect (msg_index, part_index, function_call_output_dict) for all function_call_output entries
    func_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "function_call_output":
                    func_results.append((msg_idx, part_idx, part))

    if len(func_results) <= KEEP_RECENT:
        return  # 数量未超出，无需压缩
    func_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "function_call":
                        func_name_map[block.id] = block.name
    # 2. 对超出保留窗口的旧输出进行截断（原地修改 dict）
    to_compress = func_results[:-KEEP_RECENT]
    for _, _, part in to_compress:
        output = part.get("content", "")
        if isinstance(output, str) and len(output) > 100:
            func_id = part.get("call_id")
            func_name = func_name_map.get(func_id, "unknown")
            part["content"] = f"[Previous: used {func_name}]"
    return messages

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
    # 使用 client.responses.create（Responses API），而非旧版 client.messages.create
    conversation_text = json.dumps(messages, default=str)[:80_000]
    summary_response = client.responses.create(
        model=MODEL,
        instructions="You are a conversation summarizer. Be concise and factual.",
        input=[{
            "role": "user",
            "content": (
                "Summarize this agent conversation for continuity. Include:\n"
                # "1) What was accomplished\n"
                # "2) Current working state\n"
                # "3) Key decisions and file changes made\n\n"
                + conversation_text
            ),
        }],
        max_output_tokens=2000,
    )
    summary = summary_response.output_text  # Responses API 快捷属性

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
    "query":        lambda **kw: query_acs_data(**kw),
    "get_column_names":   lambda **kw: get_column_names(**kw),
    "compact":     lambda **kw: "__COMPACT__",   # 特殊标记，在 agent_loop 中处理 Manual compression requested.
}

# ══════════════════════════════════════════════════════════════════════════════
# 工具定义 (Tool Definitions for Responses API)
# ══════════════════════════════════════════════════════════════════════════════
# 【Responses API 工具格式】扁平结构，与 Chat Completions 不同：
#   Chat Completions: {"type":"function", "function":{"name":..., "parameters":...}}
#   Responses API:    {"type":"function", "name":..., "parameters":...}  ← 无嵌套

TOOL_MODELS = [GetColumnNames, QueryACSData]
def generate_tools(response_model: BaseModel)->dict: 
    schema = response_model.model_json_schema()
    tool = {
                "type": "function",
                "name": schema['title'], # 工具名字
                "description": schema['description'], # 工具描述
                "parameters": {
                    "type": "object",
                    "properties": schema['properties'], # 参数说明
                    "required": schema.get('required', []), # 必须要传的参数
                },
            }
    return tool

TOOLS = [generate_tools(tool_model) for tool_model in TOOL_MODELS] # 使用BaseModel自动生成
# 添加 compact 工具
TOOLS.append({"type": "function", "name": "compact","description": "Trigger manual conversation compression.",
    "parameters": {"type": "object","properties": {"focus": {"type": "string","description": "What to preserve in the summary"}}}}
    )


# ══════════════════════════════════════════════════════════════════════════════
# Agent 主循环 (Agent Loop)
# ══════════════════════════════════════════════════════════════════════════════

def agent_loop(input_messages: list, model: str = None) -> None:
    """
    Responses API 的核心 Agentic Loop。

    【循环逻辑】
    ┌──────────────────────────────────────────────────────────────┐
    │  while True:                                                 │
    │    0. 上下文压缩检查（micro_compact + 超阈值则 auto_compact）│
    │    1. client.responses.create → response                    │
    │    2. input_messages += response.output  (维持上下文)        │
    │    3. 检查 response.output 中是否有 tool_call 项        │
    │       - 无 → 返回 response.output_text，结束循环             │
    │       - 有 → dispatch 执行，追加 tool_result，继续  │
    └──────────────────────────────────────────────────────────────┘

    参数:
        input_messages: 上下文列表，每轮循环后原地扩展。
                        父/子 Agent 各自持有独立实例。
        model:          模型名称，默认使用全局 MODEL。
    返回:
        无# 模型对input_messages做修改，用户自己维护。
    """
    _model = model or MODEL

    while True:
        # ── 步骤 0：上下文压缩 ──────────────────────────────────────────────
        # Layer 1：微压缩，截断旧工具输出（每次循环都执行，开销极低）
        micro_compact(input_messages)

        # Layer 2：自动压缩，token 过多时 LLM 摘要替换整段历史
        if estimate_tokens(input_messages) > COMPACT_THRESHOLD:
            print("[auto_compact triggered]")
            input_messages[:] = auto_compact(input_messages)

        # ── 步骤 1：调用 Responses API ──────────────────────────────────────
        # instructions : 系统级指令，独立于对话历史（Responses API 专有参数）
        # input        : 完整上下文列表（包含用户消息、历史模型输出、工具结果）
        # tools        : 可调用工具列表（Responses API 扁平格式）
        response = client.responses.create(
            model=_model,
            instructions=SYSTEM,
            input=input_messages,
            tools=TOOLS,
        )

        # ── 步骤 2：将模型输出追加到上下文 ─────────────────────────────────
        # response.output 是列表，可能包含：
        #   ResponseOutputMessage    (type="message")       → 最终文字回复
        #   ResponseFunctionToolCall (type="function_call") → 工具调用请求
        # 追加后，下一轮模型能看到自己上轮的输出，保证多轮连贯性。
        input_messages.append({"role": "assistant", "content": response.output_text})
        # ── 步骤 3：检查是否有工具调用 ─────────────────────────────────────
        function_calls = [
            item for item in response.output
            if hasattr(item, "type") and item.type == "function_call"
        ]
        if not function_calls:
            # 无工具调用 → 模型已给出最终回答，返回
            print("程序运行到了这里")
            return 

        # ── 步骤 4：通过 dispatch 执行工具，收集结果 ───────────────────────
        manual_compact = False
        results = []
        for item in function_calls:
            # item.name      : 工具名（对应 TOOLS 中的 "name"）
            # item.arguments : JSON 字符串（对应 "parameters"）
            if item.name == "compact":
                manual_compact = True
                output = "Compressing..."
            else: 
                handler = TOOL_HANDLERS.get(item.name)
                try:
                    output = handler(**json.loads(item.arguments)) if handler else f"Unknown tool: {item.name}"
                except Exception as e:
                    output = f"Error: {e}"
            
            print(f"> {item.name}: {str(output)[:200]}")
            results.append({"type": "function_call_output", "call_id": item.call_id, "content": str(output)})
        input_messages.append({"role": "user", "content": results})
        # Layer 3: manual compact triggered by the compact tool 模型手动触发压缩（通过 compact 工具）
        if manual_compact: 
            print("[manual compact]")
            input_messages[:] = auto_compact(input_messages)


if __name__ == "__main__":
    ## smoke test 
    user_query = "数据库中有哪些表"
    history = [{"role": "user", "content": user_query}]
    agent_loop(history)
    print(history)
    # print(history[-1]["content"])
