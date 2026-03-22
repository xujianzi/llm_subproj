from typing import List, Dict
from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    normalized = []
    for r in results:
        normalized.append(
            {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
        )
    return normalized
