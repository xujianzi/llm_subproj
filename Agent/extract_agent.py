
from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
from dotenv import load_dotenv
import sys
from pydantic import BaseModel, Field
from typing import List, Literal, Type
from template import TEXT_SYSTEM_PROMPT, get_text_user_prompt, NER_SYSTEM_PROMPT, build_ner_prompt

load_dotenv()

# ── tuneable parameters ──────────────────────────────────────────────────────
MODEL_NAME      = os.environ.get("LLM_MODEL_QWEN_FLASH", "gpt-4o-mini")
BASE_URL   = os.environ.get("DASHSCOPE_BASE_URL", None)
API_KEY    = os.environ.get("OPENAI_API_KEY", None)

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


def construct_usr_prompt(text: str) -> str:
    """构造 user 角色消息，将原始文本填入模板。"""
    return get_text_user_prompt(text)

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
    ### test  1
    # agent1 = AgentforExtraction(MODEL_NAME, build_client())
    # raw_text = "无尽冬日是一款好玩的游戏" # 张三在周日准备去北京玩
    # prompt = construct_usr_prompt(raw_text)
    # result = agent1.call(prompt, Text, TEXT_SYSTEM_PROMPT)
    # print(result)
    ### test 2
    agent2 = AgentforExtraction(MODEL_NAME, build_client())
    raw_text = "张三在周日准备去北京玩" # 张三在周日准备去北京玩
    prompt = build_ner_prompt(raw_text)
    result = agent2.call(prompt, NERLabel, NER_SYSTEM_PROMPT)
    print(result)