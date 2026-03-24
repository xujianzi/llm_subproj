#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""

"""

import json
import os
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL_NAME = "qwen-plus"
BASE_URL   = os.getenv("DASHSCOPE_BASE_URL")
API_KEY    = os.getenv("OPENAI_API_KEY")

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
MODEL = MODEL_NAME

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS, max_tokens=8000,
        )
        msg = response.choices[0].message
        # Append assistant turn
        messages.append(msg)
        # If the model didn't call a tool, we're done
        if response.choices[0].finish_reason != "tool_calls":
            return
        # Execute each tool call, collect results
        for tc in msg.tool_calls or []:
            args = json.loads(tc.function.arguments)
            print(f"\033[33m$ {args['command']}\033[0m")
            output = run_bash(args["command"])
            print(output[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]
        # last may be a ChatCompletionMessage object or a tool result dict
        content = last.content if hasattr(last, "content") else last.get("content", "")
        if content:
            print(content)
        print()
