from pathlib import Path
from typing import Dict
from loguru import logger
from extraction_agent import extract_fields
from innovation_agent import evaluate_innovation
from market_agent import analyze_market
from compliance_agent import compliance_check
from websearch_agent import search_web
from config import get_settings
from rag import get_vector_store, similarity_search
from utils import load_file_as_text


def analyze_project(project_id: str, web_search: bool = True, top_k: int = 8, force_refresh: bool = False) -> Dict:
    project_dir = Path("storage/uploads") / project_id
    if not project_dir.exists():
        raise FileNotFoundError(f"Project {project_id} not found. Please ingest first.")

    texts = [load_file_as_text(p) for p in project_dir.glob("*")]
    fields = extract_fields(texts)

    web_insights = search_web(_make_query(fields), max_results=top_k) if web_search else []

    innovation = evaluate_innovation(fields, web_insights)
    market = analyze_market(fields, web_insights)
    compliance = compliance_check(fields, top_k=top_k)
    regulation_refs = compliance["references"]

    summary = {
        "project_id": project_id,
        "innovation": innovation,
        "market": market,
        "compliance": compliance["tasks"],
        "regulation_refs": regulation_refs,
        "web_insights": web_insights[:5],
        "fields": fields,
    }
    logger.info("Analysis complete", summary=summary)
    return summary


def _make_query(fields: Dict) -> str:
    industry = fields.get("industry", "")
    tech = fields.get("tech", "")
    return f"{industry} {tech} 竞品 创新 同类产品"
