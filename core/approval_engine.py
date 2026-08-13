from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "approval_queue.json"
SENSITIVE = {"publish", "financial_plan", "external_write", "github_push", "delete", "system_change"}


def _load() -> Dict[str, Any]:
    if not STORE.is_file():
        return {"requests": {}}
    try:
        value = json.loads(STORE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"requests": {}}
    except (OSError, json.JSONDecodeError):
        return {"requests": {}}


def _save(value: Dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    temp = STORE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STORE)


def request_approval(params: Dict[str, Any]) -> Dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    if action not in SENSITIVE:
        return {"ok": False, "local_only": True, "error": "sensitive_action_required", "allowed_actions": sorted(SENSITIVE)}
    queue = _load()
    request_id = "apr_" + secrets.token_hex(8)
    record = {"id": request_id, "action": action, "payload": params.get("payload", {}), "status": "pending", "created_at": datetime.now(timezone.utc).isoformat(), "decided_at": None, "decision_note": ""}
    queue.setdefault("requests", {})[request_id] = record
    _save(queue)
    return {"ok": True, "local_only": True, "requires_human_review": True, "approval": record}


def get_approval(params: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(params.get("request_id", "")).strip()
    record = _load().get("requests", {}).get(request_id)
    if not record:
        return {"ok": False, "local_only": True, "error": "approval_not_found"}
    return {"ok": True, "local_only": True, "approval": record}


def decide_approval(params: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(params.get("request_id", "")).strip()
    decision = str(params.get("decision", "")).strip().lower()
    if decision not in {"approved", "rejected"}:
        return {"ok": False, "local_only": True, "error": "decision_must_be_approved_or_rejected"}
    queue = _load()
    record = queue.setdefault("requests", {}).get(request_id)
    if not record:
        return {"ok": False, "local_only": True, "error": "approval_not_found"}
    if record.get("status") != "pending":
        return {"ok": False, "local_only": True, "error": "approval_already_decided", "status": record.get("status")}
    record["status"] = decision
    record["decided_at"] = datetime.now(timezone.utc).isoformat()
    record["decision_note"] = str(params.get("note", ""))[:1000]
    _save(queue)
    return {"ok": True, "local_only": True, "approved": decision == "approved", "approval": record}


_HANDLERS = {"approval.request": request_approval, "approval.status": get_approval, "approval.decide": decide_approval}


def dispatch(operation: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        fn = _HANDLERS.get(operation)
        if fn is None:
            return {"ok": False, "local_only": True, "error": "unknown_operation", "operation": operation}
        return fn(params or {})
    except (OSError, TypeError, ValueError) as exc:
        return {"ok": False, "local_only": True, "error": type(exc).__name__, "detail": str(exc)}
