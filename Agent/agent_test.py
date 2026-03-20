from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
from dotenv import load_dotenv
import sys
from pydantic import BaseModel, Field
from typing import List, Literal, Type

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

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "Ticket",
#             "description": "根据用户提供的信息查询火车时刻",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "date": {
#                         "description": "要查询的火车日期",
#                         "title": "Date",
#                         "type": "string",
#                     },
#                     "departure": {
#                         "description": "出发城市或车站",
#                         "title": "Departure",
#                         "type": "string",
#                     },
#                     "destination": {
#                         "description": "要查询的火车日期",
#                         "title": "Destination",
#                         "type": "string",
#                     },
#                 },
#                 "required": ["date", "departure", "destination"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "Search",
#             "description": "根据用户提供的信息查询相关内容",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {
#                         "description": "要查询的内容",
#                         "title": "Query",
#                         "type": "string",
#                     },
#                 },
#                 "required": ["query"],
#             },
#         },
#     }
# ]

# messages = [
#     # {"role": "user", "content": "查询2024年1月1日从北京到上海的火车时刻"},
#     {"role": "user", "content": "今天的天气如何"},
# ]

# client = build_client()
# response = client.chat.completions.create(
#     model=MODEL,
#     messages=messages,
#     tools=tools,
#     tool_choice="auto",
# )
# print(response.choices[0].message.tool_calls)
# print(response.choices[0].message.tool_calls[0].function)

#-------------------------- 使用basemodel-----------------------------

# class Text(BaseModel):
#     """文本问答内容解析"""
#     search: bool = Field(description="是否需要搜索")
#     keywords: List[str] = Field(description="待选关键词")
#     intent: Literal['music', 'app', 'weather'] = Field(description="意图")

# print(Text.model_json_schema())   # 把 BaseModel → 转成 JSON Schema（标准结构说明书）
# print(Text.model_json_schema()['required'])

# arguments = '{"search": true, "keywords": ["周杰伦"], "intent": "music"}'

# data = Text.model_validate_json(arguments)  # 把 JSON 字符串 → 转成 BaseModel 对象（并自动校验）
# print(type(data))
# print(data)

# data = Text(
#     search=True,
#     keywords=["周杰伦"],
#     intent="music"
# )

# handle_text(**data.model_dump())  # 使用 .model_dump() + ** 解包传入func


"""
{
  "properties": {
    "user_name": {
      "title": "User Name",
      "type": "string"
    }
  }
}
Pydantic 默认会：
把字段名 → 转成“更友好的标题格式
search      → Search
user_name   → User Name
created_at  → Created At
"""

#----------------------- 将agent模块化，自动生成tools的json,实现信息抽取
client = build_client()
class AgentforExtraction:
    def __init__(self, model_name:str) -> None:
        self.model_name = model_name 
    
    def call(self, user_prompt:str, response_model:Type[BaseModel]):
        messages = [
            {"role":"user","content": user_prompt}
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
        response = client.chat.completions.create(
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

    def call_multi(self, user_prompt: str, response_models: List[Type[BaseModel]]):
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
        messages = [{"role": "user", "content": user_prompt}]
        response = client.chat.completions.create(
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

class Text(BaseModel):
    """文本问答内容解析"""
    search: bool = Field(description="是否需要搜索")
    keywords: List[str] = Field(description="待选关键词")
    intent: Literal['music', 'app', 'weather', 'car'] = Field(description="意图")

# result = AgentforExtraction(MODEL_NAME).call("汽车发动机和轮胎出故障了，如何处理？", Text)
# print(result)

class Ticket(BaseModel):
    """根据用户提供的信息查询火车时刻"""
    date: str = Field(description="要查询的火车日期")
    departure: str = Field(description="出发城市或车站")
    destination: str = Field(description="要查询的火车日期")

# result = AgentforExtraction(MODEL_NAME).call("你能帮我查一下2024年1月1日从北京南站到上海的火车票吗？",Ticket)
# print(result)

class Textlcut(BaseModel):
    """抽取句子中的的单词，进行文本分词"""
    keyword: List[str] = Field(description="单词")
# result = AgentforExtraction(model_name = "qwen-plus").call('小强是小王的好朋友。谢大脚是长贵的老公。', Textlcut)
# print(result)
# results = AgentforExtraction(MODEL_NAME).call_multi("你能帮我查一下2024年1月1日从北京南站到上海的火车票吗？", [Textlcut, Ticket])
# for i in results:
#     print(i)

class Text(BaseModel):
    """抽取句子的摘要"""
    abstract: str = Field(description="摘要结果")
# result = AgentforExtraction(model_name = "qwen-plus").call("20年来，中国探月工程从无到有、从小到大、从弱到强。党的十八大后，一个个探月工程任务连续成功，不断刷新世界月球探测史的中国纪录嫦娥三号实现我国探测器首次地外天体软着陆和巡视探测，总书记肯定“在人类攀登科技高峰征程中刷新了中国高度”；", Text)
# print(result)

class Text(BaseModel):
    """抽取实体"""
    person: List[str] = Field(description="人名")
    location: List[str] = Field(description="地名")
# result = AgentforExtraction(model_name = "qwen-plus").call('小明准备于2024年1月1日搭乘火车从北京南站到上海吗', Text)
# print(result)
results = AgentforExtraction(model_name = "qwen-plus").call_multi("小明准备于2024年1月1日搭乘火车从北京南站到上海", [Text, Ticket])
for i in results:
    print(f"Tool: {type(i).__name__}")   # 调用的是哪个 tool（类名）    
    print(f"Data: {i}")                  # 内容                                                                                                                                                                                  print(f"Data: {i}")                  # 内容                                                                                                                                                                                                  
    print("*" * 10)                                                                                                                                                                                                                              
                           