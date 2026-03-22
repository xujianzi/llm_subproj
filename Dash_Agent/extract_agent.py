
from openai import OpenAI
from openai.types.chat import ChatCompletion
import sys
from pydantic import BaseModel, Field
from typing import List, Literal, Type

from Settings import get_settings
from model_registry import resolve_model

_cfg = get_settings()

# ── tuneable parameters ──────────────────────────────────────────────────────
MODEL_NAME = resolve_model("qwen-plus")
BASE_URL   = _cfg.dashscope_base_url
API_KEY    = _cfg.openai_api_key

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


class AgentforExtraction:
    def __init__(self, model_name:str, client:OpenAI) -> None:
        self.model_name = model_name
        self.client = build_client() 
    
    def call(self, user_prompt:str, response_model:Type[BaseModel], system_prompt:str = ""):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        tools = [
            {
                "type": "function",
                    "function": {
                        "name": response_model.model_json_schema()['title'], # 工具名字
                        "description": response_model.model_json_schema()['description'], # 工具描述
                        "parameters": {
                            "type": "object",
                            "properties": response_model.model_json_schema()['properties'], # 参数说明
                            "required": response_model.model_json_schema()['required'], # 必须要传的参数
                        },
                    }
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools = tools,
            tool_choice="auto"
        )
        print(response.model)
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print("ERROR", response.choices[0].message)
            return None

    def call_multi(self, user_prompt: str, response_models: List[Type[BaseModel]], system_prompt: str = ""):
        """传入多个 BaseModel，每个对应一个 tool；返回模型实际调用的若干个 tool 解析后的实例
        return List[BaseModel]
        """
        model_map = {m.model_json_schema()['title']: m for m in response_models}
        tools = [
            {
                "type": "function",
                "function": {
                    "name": m.model_json_schema()['title'],
                    "description": m.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": m.model_json_schema()['properties'],
                        "required": m.model_json_schema().get('required', []),
                    },
                }
            }
            for m in response_models
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        print(response.model)
        try:
            tool_calls = response.choices[0].message.tool_calls
            results = []
            for tc in tool_calls:
                name = tc.function.name
                arguments = tc.function.arguments
                matched_model = model_map.get(name)
                if matched_model is None:
                    print(f"WARNING: 未知 tool name: {name}")
                    continue
                results.append(matched_model.model_validate_json(arguments))
            return results
        except Exception as e:
            print("ERROR", e, response.choices[0].message)
            return None

    def run(
        self,
        user_prompt: str,
        response_models: List[Type[BaseModel]],
        executor,
        system_prompt: str = "",
        max_turns: int = 10,
    ) -> str:
        """Agent loop：持续调用 LLM，将 tool 结果喂回，直到 LLM 停止调用 tool。

        Parameters
        ----------
        user_prompt    : 用户问题
        response_models: 可用的 tool BaseModel 列表
        executor       : 接收 BaseModel 实例、返回执行结果的函数（即 db_tools.execute）
        system_prompt  : 系统提示
        max_turns      : 最大循环轮数，防止死循环

        Returns
        -------
        LLM 最终的自然语言回复（str）
        """
        import json

        model_map = {m.model_json_schema()['title']: m for m in response_models}
        tools = [
            {
                "type": "function",
                "function": {
                    "name": m.model_json_schema()['title'],
                    "description": m.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": m.model_json_schema()['properties'],
                        "required": m.model_json_schema().get('required', []),
                    },
                }
            }
            for m in response_models
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            print(f"[turn {turn + 1}] finish_reason={finish_reason}")

            # LLM 不再调用 tool，返回最终回复
            if finish_reason == "stop" or not msg.tool_calls:
                return msg.content or ""

            # 将 LLM 的 tool call 消息追加到上下文
            messages.append(msg)

            # 逐个执行 tool，将结果以 tool role 喂回
            for tc in msg.tool_calls:
                name = tc.function.name
                matched_model = model_map.get(name)
                if matched_model is None:
                    tool_output = f"ERROR: unknown tool {name}"
                else:
                    instance = matched_model.model_validate_json(tc.function.arguments)
                    print(f"  [exec] {name}({tc.function.arguments})")
                    try:
                        result = executor(instance)
                        tool_output = json.dumps(result, ensure_ascii=False, default=str)
                    except Exception as e:
                        tool_output = f"ERROR: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })

        return "ERROR: max_turns reached"

class Text(BaseModel):
    """抽取实体"""
    person: List[str] = Field(description="人名")
    location: List[str] = Field(description="地名")

class NERLabel(BaseModel):
    """按字符级返回NER BIO标注结果"""
    text: str = Field(description="输入原文")
    labels: List[str] = Field(
        description=(
            "与输入文本按字符一一对应的BIO标签序列。"
            "可选标签包括：O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG"
        )
    )


if __name__ == "__main__":
    ## test  1
    agent1 = AgentforExtraction(MODEL_NAME, build_client())
    raw_text = "无尽冬日是一款好玩的游戏" # 张三在周日准备去北京玩
    prompt = construct_usr_prompt(raw_text)
    result = agent1.call(prompt, Text, TEXT_SYSTEM_PROMPT)
    print(result)
