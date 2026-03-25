"""web/backend/routers/chat_router.py — POST /api/chat/stream (SSE)."""
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from services.chat_service import chat_stream

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


@router.post("/stream")
async def stream(request: ChatRequest):
    async def generator():
        async for event in chat_stream(
            message=request.message,
            history=[m.model_dump() for m in request.history],
        ):
            yield event

    return EventSourceResponse(generator())
