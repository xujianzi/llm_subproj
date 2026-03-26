"""web/backend/main.py — FastAPI application entry point."""
import sys
from pathlib import Path

# Inject parent repo root so we can import agent.py, query_db.py, etc.
_root = str(Path(__file__).resolve().parent.parent.parent)
_backend = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
# Inject backend dir so routers can do `from services.xxx import ...`
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers.map_router import router as map_router
from .routers.chat_router import router as chat_router

app = FastAPI(title="ACS Data Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(map_router, prefix="/api/map")
app.include_router(chat_router, prefix="/api/chat")


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve simple frontend — must be mounted last
_static = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
