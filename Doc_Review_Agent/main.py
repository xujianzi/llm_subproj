import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

from config import get_settings
from orchestrator import analyze_project
from ingestion import ingest_documents, ingest_regulation_docs


app = FastAPI(title="Doc Review Multi-Agent", version="0.1.0")
settings = get_settings()

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AnalyzeRequest(BaseModel):
    project_id: str
    force_refresh: bool = False
    web_search: bool = True
    top_k: int = 8


@app.post("/ingest")
async def ingest(project_id: str, files: List[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        dest = UPLOAD_DIR / project_id / f.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)
        saved_paths.append(dest)
    try:
        ingest_documents(project_id, saved_paths)
    except Exception as exc:  # pragma: no cover - surfaced as HTTP error
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"project_id": project_id, "files": [p.name for p in saved_paths]}


@app.post("/ingest/regulation")
async def ingest_regulations(files: List[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        dest = UPLOAD_DIR / "regulation" / f.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)
        saved_paths.append(dest)
    try:
        ingest_regulation_docs(saved_paths)
    except Exception as exc:  # pragma: no cover
        logger.exception("Regulation ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"files": [p.name for p in saved_paths]}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        report = analyze_project(
            project_id=req.project_id,
            web_search=req.web_search and settings.search_enabled,
            top_k=req.top_k,
            force_refresh=req.force_refresh,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(report)


@app.get("/status")
def status():
    return {"status": "ok", "env": settings.env}
