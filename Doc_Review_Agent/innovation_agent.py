from typing import List, Dict


def evaluate_innovation(fields: Dict, web_insights: List[Dict]) -> Dict:
    """Very lightweight heuristic innovation assessment."""
    signals = []
    score = 0.5
    if fields.get("tech"):
        score += 0.1
        signals.append("提供明确技术路线")
    if len(web_insights) > 3:
        score -= 0.05
        signals.append("市场已有多家同类产品")
    else:
        score += 0.05
        signals.append("同类公开资料较少，存在空白")
    if "专利" in fields.get("raw", ""):
        score += 0.05
        signals.append("提到专利/原创算法")

    score = max(0.0, min(1.0, score))
    summary = (
        "项目具有一定创新性，需进一步验证差异化价值"
        if score >= 0.6
        else "创新性一般，建议补充差异化论证"
    )
    return {"score": score, "summary": summary, "signals": signals}
