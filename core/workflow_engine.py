from __future__ import annotations

from typing import Any, Dict

from .approval_engine import dispatch as approval
from .content_engine import dispatch as content
from .monetization_engine import dispatch as monetization
from .research_engine import dispatch as research


def adsense_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(params.get("topic", "")).strip()
    if not topic:
        return {"ok": False, "local_only": True, "error": "topic_required"}
    audience = str(params.get("audience", "developers")).strip()
    sources = params.get("documents", []) or []
    snapshots = params.get("snapshots", []) or []
    opportunity = monetization("monetization.score_opportunity", {"topic": topic, "audience": audience, "competition": params.get("competition", 0.5), "commercial_intent": params.get("commercial_intent", 0.7), "evergreen": params.get("evergreen", 0.8)})
    intelligence = research("research.intelligence_report", {"documents": sources, "snapshots": snapshots})
    brief = monetization("monetization.content_brief", {"topic": topic, "audience": audience, "intent": "commercial investigation"})
    draft = content("content.generate_draft", {"topic": topic, "audience": audience, "search_intent": "commercial investigation", "keywords": params.get("keywords", [])})
    validation = content("content.validate_draft", {"draft": draft.get("draft", {})})
    policy = monetization("monetization.policy_check", {"title": draft.get("draft", {}).get("title", ""), "body": " ".join(x.get("text", "") for x in draft.get("draft", {}).get("sections", [])), "disclosures": ["draft only; no publishing performed"]})
    metrics = monetization("monetization.metrics", params.get("metrics", {}))
    approval_request = approval("approval.request", {"action": "publish", "payload": {"topic": topic, "publication_state": "draft", "policy": policy, "validation": validation}})
    ok = all(x.get("ok", False) for x in [opportunity, intelligence, brief, draft, validation, policy, metrics, approval_request])
    return {"ok": ok, "local_only": True, "publication_performed": False, "workflow": {"opportunity": opportunity, "intelligence": intelligence, "brief": brief, "draft": draft, "validation": validation, "policy": policy, "metrics": metrics, "approval": approval_request}, "next_required_action": "human_approval_before_publish"}


_HANDLERS = {"workflow.adsense_simulation": adsense_simulation}


def dispatch(operation: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    fn = _HANDLERS.get(operation)
    if fn is None:
        return {"ok": False, "local_only": True, "error": "unknown_operation", "operation": operation}
    try:
        return fn(params or {})
    except (TypeError, ValueError, OSError) as exc:
        return {"ok": False, "local_only": True, "error": type(exc).__name__, "detail": str(exc)}
