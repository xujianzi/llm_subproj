import re
from typing import Dict, List
from loguru import logger


def extract_fields(texts: List[str]) -> Dict:
    """Lightweight regex/heuristic extractor for key proposal fields."""
    joined = "\n".join(texts)
    budget = _extract_budget(joined)
    milestones = _extract_section(joined, ["里程碑", "进度", "时间表"])
    tech = _extract_section(joined, ["技术方案", "技术路线", "架构"])
    objectives = _extract_section(joined, ["目标", "目的", "愿景"])
    industry = _guess_industry(joined)
    audience = _extract_section(joined, ["用户", "客户", "受众"])
    return {
        "budget": budget,
        "milestones": milestones,
        "tech": tech,
        "objectives": objectives,
        "industry": industry,
        "audience": audience,
        "raw": joined[:4000],
    }


def _extract_budget(text: str) -> float | None:
    # Simple currency regex: look for number + 万/亿/元
    pattern = r"([\d,.]+)\s*(万|億|亿)?\s*(?:元|人民币|RMB)?"
    matches = re.findall(pattern, text)
    if not matches:
        return None
    num_str, unit = matches[0]
    num = float(num_str.replace(",", ""))
    if unit in {"万"}:
        num *= 10_000
    if unit in {"亿", "億"}:
        num *= 100_000_000
    logger.debug(f"Budget extracted as {num}")
    return num


def _extract_section(text: str, keywords: List[str]) -> str:
    for kw in keywords:
        pattern = rf"{kw}[:：]?\s*(.+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)[:400]
    return ""


def _guess_industry(text: str) -> str:
    keywords = {
        "金融": "金融",
        "教育": "教育",
        "医疗": "医疗健康",
        "出行": "出行",
        "能源": "能源",
        "制造": "制造",
        "AI": "人工智能",
        "大模型": "人工智能",
    }
    for k, v in keywords.items():
        if k in text:
            return v
    return "未指定"
