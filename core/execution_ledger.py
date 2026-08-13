from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(os.environ.get("HERMES_ROOT", Path(__file__).resolve().parents[1]))
STORE = ROOT / "data" / "execution_ledger.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load() -> Dict[str, Any]:
    if not STORE.exists():
        return {"version": 1, "executions": [], "procedures": [], "events": []}
    return json.loads(STORE.read_text(encoding="utf-8"))


def _save(data: Dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STORE)


def _clean(value: Any, limit: int = 12000) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [_clean(v, limit) for v in value[:100]]
    if isinstance(value, dict):
        return {str(k): _clean(v, limit) for k, v in list(value.items())[:100]}
    return value


def _event(data: Dict[str, Any], operation: str, payload: Dict[str, Any]) -> None:
    data.setdefault("events", []).append({"id": uuid.uuid4().hex, "operation": operation, "payload": _clean(payload), "at": _now()})


def record_execution(params: Dict[str, Any]) -> Dict[str, Any]:
    task = str(params.get("task", "")).strip()
    if not task:
        return {"ok": False, "local_only": True, "error": "task_required"}
    steps = params.get("steps", [])
    if not isinstance(steps, list):
        return {"ok": False, "local_only": True, "error": "steps_must_be_list"}
    data = _load()
    record = {
        "id": "exec_" + uuid.uuid4().hex[:12],
        "task": _clean(task),
        "objective": _clean(params.get("objective", "")),
        "steps": _clean(steps),
        "tools": _clean(params.get("tools", [])),
        "inputs": _clean(params.get("inputs", {})),
        "outputs": _clean(params.get("outputs", {})),
        "outcome": _clean(params.get("outcome", "")),
        "errors": _clean(params.get("errors", [])),
        "corrections": _clean(params.get("corrections", [])),
        "skill_ids": _clean(params.get("skill_ids", [])),
        "artifacts": _clean(params.get("artifacts", [])),
        "approved": bool(params.get("approved", False)),
        "local_only": True,
        "internal_reasoning_excluded": True,
        "created_at": _now(),
    }
    digest_source = json.dumps(record, ensure_ascii=False, sort_keys=True)
    record["digest"] = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
    data.setdefault("executions", []).append(record)
    _event(data, "ledger.record_execution", {"execution_id": record["id"], "task": task})
    _save(data)
    return {"ok": True, "local_only": True, "execution": record}


def promote_procedure(params: Dict[str, Any]) -> Dict[str, Any]:
    execution_id = str(params.get("execution_id", "")).strip()
    name = str(params.get("name", "")).strip()
    if not execution_id or not name:
        return {"ok": False, "local_only": True, "error": "execution_id_and_name_required"}
    data = _load()
    execution = next((x for x in data.get("executions", []) if x.get("id") == execution_id), None)
    if not execution:
        return {"ok": False, "local_only": True, "error": "execution_not_found"}
    if not execution.get("approved"):
        return {"ok": False, "local_only": True, "error": "execution_approval_required"}
    digest = hashlib.sha256((name + "\n" + json.dumps(execution.get("steps", []), ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()
    existing = next((x for x in data.get("procedures", []) if x.get("digest") == digest), None)
    if existing:
        return {"ok": True, "local_only": True, "deduplicated": True, "procedure": existing}
    procedure = {"id": "proc_" + uuid.uuid4().hex[:12], "name": _clean(name), "steps": execution.get("steps", []), "tools": execution.get("tools", []), "source_execution": execution_id, "digest": digest, "created_at": _now(), "state": "approved"}
    data.setdefault("procedures", []).append(procedure)
    _event(data, "ledger.promote_procedure", {"procedure_id": procedure["id"], "execution_id": execution_id})
    _save(data)
    return {"ok": True, "local_only": True, "deduplicated": False, "procedure": procedure}


def list_ledger(params: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    query = str(params.get("query", "")).strip().lower()
    result = {"executions": data.get("executions", []), "procedures": data.get("procedures", [])}
    if query:
        result = {k: [x for x in v if query in json.dumps(x, ensure_ascii=False).lower()] for k, v in result.items()}
    return {"ok": True, "local_only": True, "ledger": result, "counts": {k: len(v) for k, v in result.items()}, "internal_reasoning_excluded": True}


def status(params: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    return {"ok": True, "local_only": True, "path": str(STORE), "counts": {k: len(data.get(k, [])) for k in ("executions", "procedures", "events")}, "external_sync": False, "model_weights_exported": False, "internal_reasoning_excluded": True}


_HANDLERS = {"ledger.record_execution": record_execution, "ledger.promote_procedure": promote_procedure, "ledger.list": list_ledger, "ledger.status": status}


def dispatch(operation: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    handler = _HANDLERS.get(operation)
    if handler is None:
        return {"ok": False, "local_only": True, "error": "unknown_operation", "operation": operation}
    try:
        return handler(params or {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ok": False, "local_only": True, "error": type(exc).__name__, "detail": str(exc)}


def operations() -> list[str]:
    return sorted(_HANDLERS)


if __name__ == "__main__":
    print(json.dumps(status({}), ensure_ascii=False, indent=2))
