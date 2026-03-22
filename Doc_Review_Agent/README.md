# Doc Review Agent

FastAPI multi-agent system for project proposal review, innovation & market analysis, and compliance guidance.

## Quickstart
1. Install dependencies: `pip install -r requirements.txt`
2. Run API: `uvicorn main:app --reload`
3. Ingest project docs: `POST /ingest` (multipart files, query `project_id`)
4. Ingest regulation docs: `POST /ingest/regulation`
5. Trigger analysis: `POST /analyze` with body `{"project_id": "...", "web_search": true}`
6. Check health: `GET /status`

## Docker
```
docker compose up --build
```

## Structure
- `main.py` FastAPI endpoints  
- `ingestion.py` chunking + Chroma storage  
- `orchestrator.py` main agent pipeline  
- `*_agent.py` individual agent logic  
- `rag.py` vector store + embeddings  
- `utils.py` file parsing helpers  
- `storage/` uploaded docs and Chroma DB  

## Configuration
Env vars: `CHROMA_PATH`, `EMBEDDING_MODEL`, `OPENAI_API_KEY` (optional), `SEARCH_ENABLED`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.
