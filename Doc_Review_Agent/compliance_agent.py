from typing import Dict, List
from rag import get_vector_store, similarity_search
from config import get_settings


def compliance_check(fields: Dict, top_k: int = 6) -> Dict:
    settings = get_settings()
    store = get_vector_store("regulations")
    questions = _build_questions(fields)
    hits = []
    for q in questions:
        hits.extend(similarity_search(store, q, top_k=top_k))
    tasks = _map_rules(fields)
    return {"tasks": tasks, "references": hits[:top_k]}


def _build_questions(fields: Dict) -> List[str]:
    budget = fields.get("budget")
    qs = ["立项审批流程", "资金使用规范"]
    if budget:
        qs.append(f"{budget} 元 项目审批要求")
    if "数据" in fields.get("tech", ""):
        qs.append("数据安全和隐私流程")
    return qs


def _map_rules(fields: Dict) -> List[Dict]:
    budget = fields.get("budget") or 0
    tasks = []
    if budget >= 5_000_000:
        tasks.append(
            {"name": "重大资金上会审批", "owner": "财务/投委会", "trigger": f"预算 {budget}"}
        )
    if "数据" in fields.get("tech", "") or "隐私" in fields.get("tech", ""):
        tasks.append({"name": "数据合规与隐私评审", "owner": "法务/安全", "trigger": "涉及数据处理"})
    tasks.append({"name": "合同法务审查", "owner": "法务", "trigger": "所有外部合作"})
    return tasks
