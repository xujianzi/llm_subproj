from typing import Dict, List


def analyze_market(fields: Dict, web_insights: List[Dict]) -> Dict:
    customers = fields.get("audience") or "待补充"
    industry = fields.get("industry")
    size_hint = "中等" if len(web_insights) > 5 else "早期/小众"
    revenue_model = "订阅+增值服务"
    cost_factors = ["研发人力", "云算力/数据成本", "市场推广"]

    return {
        "customers": customers,
        "industry": industry,
        "size_hint": size_hint,
        "revenue_model": revenue_model,
        "cost_factors": cost_factors,
        "risks": ["需求验证不足", "竞争加剧", "政策合规风险"],
    }
