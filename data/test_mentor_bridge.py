import json
import sys
sys.path.insert(0, r"D:\Hermes")
from core.mentor_bridge import dispatch

lesson = dispatch("bridge.record_lesson", {
    "title": "قاعدة التحقق قبل اعتماد مهارة",
    "lesson": "لا تتحول أي ملاحظة أو خبرة إلى معرفة معتمدة قبل وجود دليل قابل للتدقيق واختبار ناجح واعتماد صريح.",
    "evidence": ["local_test", "human_review_gate"],
    "tags": ["governance", "testing"],
    "source": "mentor_session"
})
assert lesson["ok"] and lesson["local_only"]
lesson_id = lesson["lesson"]["id"]
blocked = dispatch("bridge.promote_lesson", {"lesson_id": lesson_id, "approval_ref": "apr_missing_for_test"})
assert blocked["ok"] is False and blocked["error"] == "approval_not_approved"
from core.approval_engine import dispatch as approval_dispatch
approval_request = approval_dispatch("approval.request", {"action": "publish", "payload": {"lesson_id": lesson_id}})
assert approval_request["ok"] and approval_request["requires_human_review"]
approval_id = approval_request["approval"]["id"]
approval_decision = approval_dispatch("approval.decide", {"request_id": approval_id, "decision": "approved", "note": "Local human approval test"})
assert approval_decision["ok"] and approval_decision["approved"]
approved = dispatch("bridge.promote_lesson", {"lesson_id": lesson_id, "approval_ref": approval_id})
assert approved["ok"] and approved["lesson"]["state"] == "approved"
skill = dispatch("bridge.create_skill", {"name": "offline_review_gate", "instructions": "افحص الدليل والاختبار والاعتماد قبل ترقية المعرفة.", "triggers": ["approve", "promote"]})
assert skill["ok"] and skill["local_only"]
feedback = dispatch("bridge.record_feedback", {"operation": "bridge.promote_lesson", "outcome": "نجح بعد اعتماد صريح", "correction": "منع الترقية عند غياب مرجع الاعتماد", "evidence": ["blocked_without_ref", "approved_with_ref"]})
assert feedback["ok"]
listing = dispatch("bridge.list_knowledge", {"query": "اعتماد"})
assert listing["ok"]
shot = dispatch("bridge.snapshot", {})
assert shot["ok"] and shot["local_only"]
status = dispatch("bridge.status", {})
assert status["ok"] and status["external_sync"] is False and status["model_weights_exported"] is False
print(json.dumps({"lesson": lesson, "blocked": blocked, "approved": approved, "skill": skill, "feedback": feedback, "listing": listing, "snapshot": shot, "status": status}, ensure_ascii=False, indent=2))
