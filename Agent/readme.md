# Extract Agent

基于 OpenAI function calling 的结构化信息抽取框架。

## 主要亮点

### 1. 动态 Tool 调用 + 结构化返回

`AgentforExtraction` 接收任意 `BaseModel` 子类作为参数，在运行时自动将其 JSON Schema 转换为 OpenAI function calling 的 tool 定义。

- 无需硬编码 tool 描述，新增抽取任务只需定义一个新的 `BaseModel`
- LLM 的返回结果直接通过 `model_validate_json` 解析回强类型对象，调用方拿到的是结构化数据而非原始字符串
- `call_multi` 支持同时注册多个 tool，让 LLM 在一次调用中按需选择并返回多个结果

```python
class Text(BaseModel):
    """抽取实体"""
    person: List[str] = Field(description="人名")
    location: List[str] = Field(description="地名")

result = agent.call(prompt, Text)   # 返回 Text 实例，而非字符串
```

### 2. Template 与 Agent 解耦，增强抽取准确性

`template.py` 为每类子任务单独维护 system prompt 和 user prompt 构造函数，通过明确的角色约束（system/user 分离）引导 LLM 行为：

- **system prompt**：静态指令，定义角色、规则、输出格式，作为 LLM 的行为约束
- **user prompt**：动态内容，仅包含待处理文本，保持职责单一
- 新增任务类型时只需在 `template.py` 添加对应模板，`AgentforExtraction` 无需改动

```python
# 切换任务只需换模板，agent 层不变
result = agent.call(construct_usr_prompt(text), Text, system_prompt=NER_SYSTEM_PROMPT)
```

### 3. 新增子任务扩展性强

框架对扩展完全开放，对修改完全关闭。创建一个新的抽取任务只需两步，无需触碰 `AgentforExtraction` 的任何代码：

1. 在 `template.py` 添加对应的 system prompt 和 user prompt 构造函数
2. 定义一个新的 `BaseModel` 描述返回结构

```python
# Step 1 - template.py
SENTIMENT_SYSTEM_PROMPT = """你是一个情感分析助手..."""
def get_sentiment_user_prompt(text: str) -> str:
    return f"待分析文本：\n{text}"

# Step 2 - 定义返回结构
class Sentiment(BaseModel):
    """情感分析"""
    label: Literal["正面", "负面", "中性"] = Field(description="情感极性")
    confidence: float = Field(description="置信度")

# 直接复用同一个 agent 实例
result = agent.call(get_sentiment_user_prompt(text), Sentiment, system_prompt=SENTIMENT_SYSTEM_PROMPT)
```

## 文件结构

| 文件 | 职责 |
|------|------|
| `extract_agent.py` | Agent 核心逻辑：client 构建、tool 注册、LLM 调用、结果解析 |
| `template.py` | 各子任务的 system/user prompt 模板 |
