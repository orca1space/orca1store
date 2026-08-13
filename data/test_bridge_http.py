import json
import sys
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:7777/api/agent/exec"

def call(op, params):
    body = json.dumps({"op": op, "params": params}, ensure_ascii=False).encode("utf-8")
    req = Request(BASE, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))

lesson = call("bridge.record_lesson", {"title": "HTTP bridge lesson", "lesson": "Knowledge is promoted only after a real local approval record.", "evidence": ["http_api_test"], "source": "http_test"})
assert lesson["ok"]
lesson_id = lesson["result"]["lesson"]["id"]
blocked = call("bridge.promote_lesson", {"lesson_id": lesson_id, "approval_ref": "apr_http_missing"})
assert blocked["ok"] and blocked["result"]["ok"] is False
request = call("approval.request", {"action": "publish", "payload": {"lesson_id": lesson_id}})
assert request["ok"] and request["result"]["requires_human_review"]
approval_id = request["result"]["approval"]["id"]
decision = call("approval.decide", {"request_id": approval_id, "decision": "approved", "note": "HTTP local human approval test"})
assert decision["ok"] and decision["result"]["approved"]
promoted = call("bridge.promote_lesson", {"lesson_id": lesson_id, "approval_ref": approval_id})
assert promoted["ok"] and promoted["result"]["ok"] and promoted["result"]["lesson"]["state"] == "approved"
status = call("bridge.status", {})
assert status["ok"] and status["result"]["external_sync"] is False
print(json.dumps({"lesson": lesson, "blocked": blocked, "approval_request": request, "decision": decision, "promoted": promoted, "status": status}, ensure_ascii=False, indent=2))
