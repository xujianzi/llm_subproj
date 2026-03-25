"""web/backend/main.py — FastAPI application entry point."""
import sys
from pathlib import Path

# Inject parent repo root so we can import agent.py, query_db.py, etc.
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.map_router import router as map_router
from .routers.chat_router import router as chat_router

app = FastAPI(title="ACS Data Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(map_router, prefix="/api/map")
app.include_router(chat_router, prefix="/api/chat")


@app.get("/health")
def health():
    return {"status": "ok"}
