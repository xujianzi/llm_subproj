from openai import OpenAI
import sys
from dotenv import load_dotenv
import os 
from openai.types.chat import ChatCompletion

load_dotenv()

MODEL = "qwen-plus"
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

messages = [{"role": "system", "content": SYSTEM}]
query = "你好"
# messages.append({"role": "user", "content": query})

# response = client.responses.create(
#     model=MODEL_NAME,
#     inputs=messages,
#     max_tokens=8000,
# )

def agent_loop(query):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query}
    ]
    while True:
        response = client.responses.create(
            model=MODEL, 
            input=messages,
            tools=TOOLS, 
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})

