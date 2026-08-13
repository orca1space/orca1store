"""Deterministic, offline monetization planning primitives for Hermes."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import re

POLICY_TERMS = {
    "invalid_traffic": ["click your ad", "click on ads", "buy traffic", "paid clicks", "bot traffic"],
    "spam": ["keyword stuffing", "doorway page", "cloaking", "link spam", "scraped content"],
    "risk": ["guaranteed income", "get rich quick", "medical cure", "financial guarantee"],
}

@dataclass
class Opportunity:
    topic: str
    audience: str
    demand: float
    competition: float
    originality: float
    production_cost: float
    policy_risk: float
    score: float
    rationale: List[str]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return default


def score_opportunity(params: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(params.get("topic", "")).strip()
    audience = str(params.get("audience", "")).strip()
    if not topic or not audience:
        return {"ok": False, "error": "topic_and_audience_required", "local_only": True}
    demand = _num(params.get("demand"), 50)
    competition = _num(params.get("competition"), 50)
    originality = _num(params.get("originality"), 50)
    cost = _num(params.get("production_cost"), 50)
    risk = _num(params.get("policy_risk"), 0)
    score = round(0.30*demand + 0.25*(100-competition) + 0.25*originality + 0.10*(100-cost) + 0.10*(100-risk), 2)
    rationale = []
    if demand >= 70: rationale.append("strong_demand_signal")
    if competition <= 35: rationale.append("manageable_competition")
    if originality >= 70: rationale.append("strong_originality")
    if risk >= 30: rationale.append("policy_review_required")
    return {"ok": True, "local_only": True, "opportunity": asdict(Opportunity(topic, audience, demand, competition, originality, cost, risk, score, rationale))}


def build_content_brief(params: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(params.get("topic", "")).strip()
    intent = str(params.get("intent", "informational")).strip()
    audience = str(params.get("audience", "general audience")).strip()
    if not topic: return {"ok": False, "error": "topic_required", "local_only": True}
    return {"ok": True, "local_only": True, "brief": {
        "topic": topic, "search_intent": intent, "audience": audience,
        "required_sections": ["original_answer", "first_hand_or_primary_sources", "methodology", "limitations", "faq", "references"],
        "quality_gates": ["people_first", "originality", "fact_checking", "citation_provenance", "policy_compliance", "human_review_before_publish"],
        "schema_candidates": ["Article", "FAQPage", "HowTo"],
        "publication_state": "draft",
    }}


def policy_check(params: Dict[str, Any]) -> Dict[str, Any]:
    text = str(params.get("text", ""))
    low = text.lower()
    findings = []
    for category, terms in POLICY_TERMS.items():
        for term in terms:
            if term in low: findings.append({"category": category, "term": term, "severity": "high" if category != "risk" else "medium"})
    return {"ok": True, "local_only": True, "publish_allowed": not any(x["severity"] == "high" for x in findings), "requires_human_review": bool(findings), "findings": findings}


def metrics_analyze(params: Dict[str, Any]) -> Dict[str, Any]:
    impressions = _num(params.get("impressions"), 0)
    clicks = _num(params.get("clicks"), 0)
    revenue = max(0.0, float(params.get("revenue", 0) or 0))
    ctr = round((clicks / impressions) * 100, 4) if impressions else 0.0
    rpm = round((revenue / impressions) * 1000, 4) if impressions else 0.0
    return {"ok": True, "local_only": True, "metrics": {"impressions": impressions, "clicks": clicks, "revenue": revenue, "ctr_percent": ctr, "rpm": rpm}, "warnings": ["small_sample"] if impressions < 1000 else []}


def create_execution_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(params.get("topic", "")).strip()
    if not topic: return {"ok": False, "error": "topic_required", "local_only": True}
    return {"ok": True, "local_only": True, "plan": [{"step": 1, "name": "research", "approval": False}, {"step": 2, "name": "brief", "approval": False}, {"step": 3, "name": "draft", "approval": False}, {"step": 4, "name": "fact_check_and_policy", "approval": False}, {"step": 5, "name": "publish", "approval": True}], "topic": topic}


def dispatch(operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"monetization.score_opportunity": score_opportunity, "monetization.content_brief": build_content_brief, "monetization.policy_check": policy_check, "monetization.metrics": metrics_analyze, "monetization.execution_plan": create_execution_plan}.get(operation, lambda _: {"ok": False, "error": "unknown_operation", "local_only": True})(params)
