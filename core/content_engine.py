from __future__ import annotations

import re
from typing import Any, Dict, List


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _keywords(params: Dict[str, Any]) -> List[str]:
    raw = params.get("keywords", [])
    if isinstance(raw, str):
        raw = re.split(r"[,\n]", raw)
    return [_clean(x) for x in raw if _clean(x)]


def generate_draft(params: Dict[str, Any]) -> Dict[str, Any]:
    topic = _clean(params.get("topic"))
    if not topic:
        return {"ok": False, "local_only": True, "error": "topic_required"}
    audience = _clean(params.get("audience")) or "general readers"
    intent = _clean(params.get("search_intent")) or "informational"
    keywords = _keywords(params)
    title = _clean(params.get("title")) or f"{topic}: a practical guide for {audience}"
    sections = params.get("sections") or ["What it is", "How it works", "Practical workflow", "Common mistakes", "FAQ", "References"]
    paragraphs = {
        "What it is": f"This guide explains {topic} for {audience}. It focuses on a practical, verifiable workflow rather than unsupported promises.",
        "How it works": f"Start by defining the intended outcome for {topic}, then document inputs, decisions, constraints, and measurable acceptance criteria.",
        "Practical workflow": f"Use a staged process for {topic}: research the available evidence, prepare a brief, create an original draft, verify claims, and obtain human approval before publication.",
        "Common mistakes": "Avoid copied or thin material, unsupported claims, repetitive keyword use, misleading headlines, undisclosed commercial relationships, and publishing without review.",
        "FAQ": f"The appropriate implementation of {topic} depends on the audience, evidence quality, operating constraints, and the review standard applied before release.",
        "References": "Add primary or first-hand sources used during verification. Keep provenance for every material factual claim."
    }
    body = []
    for section in sections:
        name = _clean(section)
        body.append({"heading": name, "text": paragraphs.get(name, f"This section covers {name.lower()} in the context of {topic}, with emphasis on evidence, limitations, and practical use.")})
    return {"ok": True, "local_only": True, "publication_state": "draft", "draft": {"title": title, "topic": topic, "audience": audience, "search_intent": intent, "keywords": keywords, "sections": body, "human_review_required": True}}


def seo_optimize(params: Dict[str, Any]) -> Dict[str, Any]:
    draft = params.get("draft") or {}
    title = _clean(draft.get("title"))
    text = " ".join(_clean(x.get("text")) for x in draft.get("sections", []) if isinstance(x, dict))
    keywords = _keywords(params) or _keywords(draft)
    missing = [kw for kw in keywords if kw.lower() not in (title + " " + text).lower()]
    return {"ok": True, "local_only": True, "analysis": {"title_present": bool(title), "word_count": len(text.split()), "keyword_count": len(keywords), "missing_keywords": missing, "recommendations": ["keep the title descriptive", "answer the search intent directly", "retain citations and limitations", "do not force keyword repetitions"]}}


def validate_draft(params: Dict[str, Any]) -> Dict[str, Any]:
    draft = params.get("draft") or {}
    findings = []
    if not _clean(draft.get("title")): findings.append({"severity": "high", "code": "missing_title"})
    sections = draft.get("sections") or []
    if len(sections) < 3: findings.append({"severity": "medium", "code": "thin_structure"})
    text = " ".join(_clean(x.get("text")) for x in sections if isinstance(x, dict)).lower()
    for term in ("guaranteed income", "click your ad", "keyword stuffing", "scraped content"):
        if term in text: findings.append({"severity": "high", "code": "policy_risk", "term": term})
    return {"ok": True, "local_only": True, "publish_allowed": not any(x["severity"] == "high" for x in findings), "requires_human_review": True, "findings": findings}


_HANDLERS = {"content.generate_draft": generate_draft, "content.seo_optimize": seo_optimize, "content.validate_draft": validate_draft}


def dispatch(operation: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        fn = _HANDLERS.get(operation)
        if fn is None:
            return {"ok": False, "local_only": True, "error": "unknown_operation", "operation": operation}
        return fn(params or {})
    except (TypeError, ValueError) as exc:
        return {"ok": False, "local_only": True, "error": type(exc).__name__, "detail": str(exc)}
