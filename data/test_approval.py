import json
import sys
sys.path.insert(0, r"D:\Hermes")
from core.approval_engine import dispatch

requested = dispatch("approval.request", {"action":"publish", "payload":{"artifact":"draft-001"}})
request_id = requested["approval"]["id"]
pending = dispatch("approval.status", {"request_id": request_id})
decided = dispatch("approval.decide", {"request_id": request_id, "decision":"approved", "note":"Human reviewer approved the simulated publication."})
replay = dispatch("approval.decide", {"request_id": request_id, "decision":"rejected"})
result = {"requested": requested, "pending": pending, "decided": decided, "replay": replay}
print(json.dumps(result, ensure_ascii=False, indent=2))
assert requested.get("requires_human_review") is True
assert pending["approval"]["status"] == "pending"
assert decided.get("approved") is True
assert replay.get("ok") is False
