from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
from pathlib import Path
import uvicorn

from agent import agent_loop

app = FastAPI()

current_dir = Path(__file__).parent
static_dir = current_dir / "static"

# 简单内存存储：生产环境建议换成 sqlite / redis / postgres
# 结构：{conversation_id: {"messages": [], "title": str}}
CONVERSATIONS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    messages: list


@app.get("/api/conversations")
def list_conversations():
    """返回所有会话的 id 和标题列表（按创建顺序）。"""
    return [
        {"conversation_id": cid, "title": data["title"]}
        for cid, data in CONVERSATIONS.items()
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or str(uuid4())

    if conversation_id not in CONVERSATIONS:
        # 用首条消息的前 30 个字符作为标题
        title = req.message[:30] + ("..." if len(req.message) > 30 else "")
        CONVERSATIONS[conversation_id] = {"messages": [], "title": title}

    history = CONVERSATIONS[conversation_id]["messages"]
    history.append({"role": "user", "content": req.message})

    agent_loop(history)

    answer = ""
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            answer = msg["content"]
            break

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        messages=history,
    )


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    data = CONVERSATIONS.get(conversation_id, {"messages": [], "title": ""})
    return {
        "conversation_id": conversation_id,
        "title": data["title"],
        "messages": data["messages"],
    }


app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8090, reload=True)
