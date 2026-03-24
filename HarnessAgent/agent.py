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

load_dotenv(override=True)

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_NAME     = "qwen-plus"
BASE_URL       = os.getenv("DASHSCOPE_BASE_URL")
API_KEY        = os.getenv("OPENAI_API_KEY")
WORKDIR        = Path(__file__).parent

# 上下文压缩阈值（估算 token 数超过此值触发 auto_compact）
COMPACT_THRESHOLD = 50_000
# micro_compact 保留最近几条工具输出不压缩
KEEP_RECENT       = 3
# 对话记录存储目录
TRANSCRIPT_DIR    = WORKDIR / ".transcripts"

# 检测操作系统，决定 shell 类型（影响 run_bash 和系统提示）
_IS_WINDOWS = sys.platform == "win32"
_SHELL_NAME = "PowerShell" if _IS_WINDOWS else "bash"


# ── 客户端 ────────────────────────────────────────────────────────────────────
def build_client() -> OpenAI:
    """构建 OpenAI 兼容客户端（支持 DashScope 等兼容 OpenAI 协议的第三方服务）。"""
    api_key = API_KEY
    if not api_key:
        sys.exit(
            "ERROR: No API key found.\n"
            "Set OPENAI_API_KEY in your environment or .env file."
        )
    kwargs: dict = {"api_key": api_key}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL   # 指向第三方兼容端点（如阿里云 DashScope）
    return OpenAI(**kwargs)


client = build_client()
MODEL  = MODEL_NAME

# ── 系统提示 ──────────────────────────────────────────────────────────────────
# Responses API 的系统指令通过独立的 instructions 参数传入，不混入 input 列表
SYSTEM = (
    f"You are a coding agent working in {WORKDIR}. "
    f"OS: {'Windows' if _IS_WINDOWS else 'Linux/macOS'}, shell: {_SHELL_NAME}. "
    "Use tools to solve tasks. Act, don't explain."
)


# ══════════════════════════════════════════════════════════════════════════════
# 上下文压缩 (Context Compression)
# ══════════════════════════════════════════════════════════════════════════════

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
        - ResponseFunctionToolCall 对象 (type="function_call") ← 模型工具调用
        - dict {"type":"function_call_output", "call_id":..., "output":...} ← 工具结果
        - ResponseOutputMessage 对象 (type="message")       ← 模型文本回复
    """
    # 1. 收集所有 function_call_output 字典的引用
    tool_outputs = [
        item for item in messages
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    if len(tool_outputs) <= KEEP_RECENT:
        return  # 数量未超出，无需压缩

    # 2. 建立 call_id → 工具名称 的映射（从 function_call 对象中提取）
    call_id_to_name: dict[str, str] = {}
    for item in messages:
        # ResponseFunctionToolCall 是 SDK 对象，有 .type / .name / .call_id 属性
        if hasattr(item, "type") and item.type == "function_call":
            call_id_to_name[item.call_id] = item.name

    # 3. 对超出保留窗口的旧输出进行截断（原地修改 dict）
    to_compress = tool_outputs[:-KEEP_RECENT]
    for result in to_compress:
        output = result.get("output", "")
        if isinstance(output, str) and len(output) > 100:
            tool_name = call_id_to_name.get(result.get("call_id", ""), "unknown")
            result["output"] = f"[Previous result: used {tool_name}]"


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
                "1) What was accomplished\n"
                "2) Current working state\n"
                "3) Key decisions and file changes made\n\n"
                + conversation_text
            ),
        }],
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
# 工具实现 (Tool Implementations)
# ══════════════════════════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """将相对路径解析为绝对路径，并强制限制在 WORKDIR 内（防目录穿越）。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR.resolve()):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    执行 shell 命令并返回输出。

    Windows 下使用 PowerShell（支持 ls/cat 等 Unix 别名），
    Linux/macOS 下使用 bash，避免 cmd.exe 不认识 Unix 命令的问题。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        if _IS_WINDOWS:
            # PowerShell 支持大多数 Unix 风格命令（ls/cat/pwd 等均有内置别名）
            args = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
            r = subprocess.run(
                args, cwd=WORKDIR,
                capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace",
            )
        else:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=120,
            )
        out = (r.stdout + r.stderr).strip()
        return out[:50_000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """读取文件内容，可选限制行数。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50_000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """将内容写入文件（自动创建目录）。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在文件中做精确字符串替换（替换首次出现）。"""
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Skill 系统 (Skill System)
# ══════════════════════════════════════════════════════════════════════════════
# Skill 是预定义的命名操作，模型通过 call_skill 工具按名称调用。
# 与普通工具的区别：Skill 封装了更高层的业务逻辑，可以组合多步操作。

SKILLS: dict[str, callable] = {
    # 代码质量类
    "git_status":    lambda: run_bash("git status"),
    "git_log":       lambda: run_bash("git log --oneline -10"),
    "list_files":    lambda: run_bash("Get-ChildItem -Name" if _IS_WINDOWS else "ls -la"),
    "show_tree":     lambda: run_bash("tree /F" if _IS_WINDOWS else "find . -not -path './.git/*' | head -50"),
    # 环境信息类
    "python_version": lambda: run_bash("python --version"),
    "show_env":       lambda: run_bash("Get-ChildItem Env:" if _IS_WINDOWS else "env | grep -v SECRET"),
}


def run_skill(skill_name: str, args: dict = None) -> str:
    """
    执行预定义 Skill。

    参数:
        skill_name: SKILLS 中注册的技能名称
        args:       可选参数（当前版本 Skill 均无参数，预留扩展）
    """
    skill = SKILLS.get(skill_name)
    if skill is None:
        available = ", ".join(SKILLS.keys())
        return f"Error: Unknown skill '{skill_name}'. Available: {available}"
    try:
        return skill()
    except Exception as e:
        return f"Error running skill '{skill_name}': {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 子 Agent (Sub-Agent)
# ══════════════════════════════════════════════════════════════════════════════

def spawn_agent(task: str, model: str = None) -> str:
    """
    创建并运行一个独立的子 Agent 来完成指定子任务。

    子 Agent 拥有独立的上下文列表（隔离，不影响父 Agent 的对话历史），
    完成任务后返回最终文本结果给父 Agent。

    参数:
        task:  子任务描述，会作为 user 消息传入子 Agent
        model: 可选，子 Agent 使用的模型（默认与父 Agent 相同）

    使用场景：
      - 长时间独立子任务（如：分析一个大文件，生成测试用例）
      - 需要与主对话隔离的探索性操作
    """
    sub_model = model or MODEL
    # 子 Agent 有独立的上下文，从零开始
    sub_messages: list = [{"role": "user", "content": task}]
    print(f"\033[35m[sub-agent] task: {task[:80]}...\033[0m")
    # 直接复用 agent_loop，传入子上下文和指定模型
    result = agent_loop(sub_messages, model=sub_model)
    print(f"\033[35m[sub-agent] done.\033[0m")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 工具派发 (Tool Dispatch)
# ══════════════════════════════════════════════════════════════════════════════
# TOOL_HANDLERS 是工具名称 → 实现函数的中央映射表（dispatch table）。
# agent_loop 通过查询此表完成 工具名 → 函数调用 的派发，无需 if/elif 链。
# 新增工具：在此添加一行 + 在 TOOLS 列表中添加定义，不需要改 agent_loop。

TOOL_HANDLERS: dict[str, callable] = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "call_skill":  lambda **kw: run_skill(kw["skill_name"], kw.get("args", {})),
    "spawn_agent": lambda **kw: spawn_agent(kw["task"], kw.get("model")),
    "compact":     lambda **kw: "__COMPACT__",   # 特殊标记，在 agent_loop 中处理
}


# ══════════════════════════════════════════════════════════════════════════════
# 工具定义 (Tool Definitions for Responses API)
# ══════════════════════════════════════════════════════════════════════════════
# 【Responses API 工具格式】扁平结构，与 Chat Completions 不同：
#   Chat Completions: {"type":"function", "function":{"name":..., "parameters":...}}
#   Responses API:    {"type":"function", "name":..., "parameters":...}  ← 无嵌套

TOOLS = [
    {
        "type": "function",
        "name": "bash",
        "description": f"Run a {_SHELL_NAME} command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file (relative to workspace).",
        "parameters": {
            "type": "object",
            "properties": {
                "path":  {"type": "string",  "description": "File path relative to workspace"},
                "limit": {"type": "integer", "description": "Max number of lines to return (optional)"},
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file, creating it or overwriting.",
        "parameters": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "File path relative to workspace"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": "Replace the first occurrence of exact text in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path":     {"type": "string", "description": "File path relative to workspace"},
                "old_text": {"type": "string", "description": "Exact text to find"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "type": "function",
        "name": "call_skill",
        "description": (
            "Execute a named skill (predefined operation). "
            f"Available skills: {', '.join(SKILLS.keys())}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Name of the skill to run"},
                "args":       {"type": "object", "description": "Optional arguments for the skill"},
            },
            "required": ["skill_name"],
        },
    },
    {
        "type": "function",
        "name": "spawn_agent",
        "description": (
            "Create an isolated sub-agent to handle an independent subtask. "
            "Returns the sub-agent's final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task":  {"type": "string", "description": "Complete description of the subtask"},
                "model": {"type": "string", "description": "Model override for the sub-agent (optional)"},
            },
            "required": ["task"],
        },
    },
    {
        "type": "function",
        "name": "compact",
        "description": "Manually trigger conversation compression to free up context space.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {"type": "string", "description": "What critical info to preserve in the summary"},
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Agent 主循环 (Agent Loop)
# ══════════════════════════════════════════════════════════════════════════════

def agent_loop(input_messages: list, model: str = None) -> str:
    """
    Responses API 的核心 Agentic Loop。

    【循环逻辑】
    ┌──────────────────────────────────────────────────────────────┐
    │  while True:                                                 │
    │    0. 上下文压缩检查（micro_compact + 超阈值则 auto_compact）│
    │    1. client.responses.create → response                    │
    │    2. input_messages += response.output  (维持上下文)        │
    │    3. 检查 response.output 中是否有 function_call 项        │
    │       - 无 → 返回 response.output_text，结束循环             │
    │       - 有 → dispatch 执行，追加 function_call_output，继续  │
    └──────────────────────────────────────────────────────────────┘

    参数:
        input_messages: 上下文列表，每轮循环后原地扩展。
                        父/子 Agent 各自持有独立实例。
        model:          模型名称，默认使用全局 MODEL。
    返回:
        模型最终文本回复（str）。
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
        input_messages += response.output

        # ── 步骤 3：检查是否有工具调用 ─────────────────────────────────────
        tool_calls = [
            item for item in response.output
            if hasattr(item, "type") and item.type == "function_call"
        ]

        if not tool_calls:
            # 无工具调用 → 模型已给出最终回答，返回文本
            return response.output_text

        # ── 步骤 4：通过 dispatch 执行工具，收集结果 ───────────────────────
        trigger_compact = False

        for item in tool_calls:
            # item.name      : 工具名（对应 TOOLS 中的 "name"）
            # item.arguments : JSON 字符串（对应 "parameters"）
            # item.call_id   : 本次调用 ID，回传结果时必须用相同 ID

            args   = json.loads(item.arguments)
            handler = TOOL_HANDLERS.get(item.name)

            if handler is None:
                output = f"Error: Unknown tool '{item.name}'"
            else:
                try:
                    output = handler(**args)
                except Exception as e:
                    output = f"Error: {e}"

            # 检测手动压缩标记
            if output == "__COMPACT__":
                trigger_compact = True
                output = "Manual compression triggered."

            # 打印执行摘要
            print(f"\033[33m[{item.name}]\033[0m {str(output)[:200]}")

            # 将工具结果追加到上下文
            # 【Responses API 格式】与 Chat Completions 完全不同：
            #   Chat Completions: {"role":"tool", "tool_call_id":..., "content":...}
            #   Responses API:    {"type":"function_call_output", "call_id":..., "output":...}
            input_messages.append({
                "type":    "function_call_output",
                "call_id": item.call_id,
                "output":  str(output),
            })

        # Layer 3：手动触发压缩（通过 compact 工具）
        if trigger_compact:
            print("[manual compact]")
            input_messages[:] = auto_compact(input_messages)


# ══════════════════════════════════════════════════════════════════════════════
# 交互入口 (Interactive Entry Point)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # input_messages 贯穿整个会话的上下文（手动维护，方式2）
    # 每轮对话：user 消息追加 → agent_loop 扩展 → 下轮继续使用
    input_messages: list = []

    print(f"\033[36mHarnessAgent ready. Model: {MODEL} | Shell: {_SHELL_NAME}\033[0m")
    print("Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query or query.lower() in ("q", "exit"):
            break

        # 追加用户消息（标准 OpenAI role/content 格式）
        input_messages.append({"role": "user", "content": query})

        # 运行 agent 循环，原地扩展 input_messages，返回最终文本
        answer = agent_loop(input_messages)
        print(f"\n{answer}\n")
