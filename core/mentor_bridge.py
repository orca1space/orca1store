from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STORE_PATH = DATA_DIR / "mentor_bridge.json"
SNAPSHOT_DIR = DATA_DIR / "mentor_snapshots"
ALLOWED_ROOTS = (ROOT.resolve(), DATA_DIR.resolve())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_path(value: Any) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not any(path == root or root in path.parents for root in ALLOWED_ROOTS):
        raise ValueError("path_outside_local_workspace")
    return path


def _load() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        return {"version": 1, "lessons": [], "skills": [], "feedback": [], "events": []}
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def _save(data: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STORE_PATH)


def _event(data: Dict[str, Any], operation: str, payload: Dict[str, Any]) -> None:
    data.setdefault("events", []).append({"id": uuid.uuid4().hex, "operation": operation, "at": _now(), "payload": payload})


def record_lesson(params: Dict[str, Any]) -> Dict[str, Any]:
    title = _clean(params.get("title"))
    lesson = _clean(params.get("lesson"))
    if not title or not lesson:
        return {"ok": False, "local_only": True, "error": "title_and_lesson_required"}
    data = _load()
    digest = hashlib.sha256((title + "\n" + lesson).encode("utf-8")).hexdigest()
    for item in data["lessons"]:
        if item.get("digest") == digest:
            return {"ok": True, "local_only": True, "deduplicated": True, "lesson": item}
    item = {"id": "lesson_" + uuid.uuid4().hex[:12], "title": title, "lesson": lesson, "evidence": params.get("evidence", []), "tags": params.get("tags", []), "source": _clean(params.get("source")) or "local_review", "state": "candidate", "digest": digest, "created_at": _now(), "updated_at": _now()}
    data["lessons"].append(item)
    _event(data, "bridge.record_lesson", {"lesson_id": item["id"]})
    _save(data)
    return {"ok": True, "local_only": True, "deduplicated": False, "lesson": item}


def promote_lesson(params: Dict[str, Any]) -> Dict[str, Any]:
    lesson_id = _clean(params.get("lesson_id"))
    approval_ref = _clean(params.get("approval_ref"))
    if not lesson_id or not approval_ref:
        return {"ok": False, "local_only": True, "error": "lesson_id_and_approval_ref_required"}
    from .approval_engine import dispatch as approval_dispatch
    approval = approval_dispatch("approval.status", {"request_id": approval_ref})
    approval_record = approval.get("approval", {}) if isinstance(approval, dict) else {}
    if not approval.get("ok") or approval_record.get("status") != "approved":
        return {"ok": False, "local_only": True, "error": "approval_not_approved", "approval": approval_record}
    data = _load()
    for item in data["lessons"]:
        if item.get("id") == lesson_id:
            item["state"] = "approved"
            item["approval_ref"] = approval_ref
            item["updated_at"] = _now()
            _event(data, "bridge.promote_lesson", {"lesson_id": lesson_id, "approval_ref": approval_ref})
            _save(data)
            return {"ok": True, "local_only": True, "lesson": item}
    return {"ok": False, "local_only": True, "error": "lesson_not_found"}


def create_skill(params: Dict[str, Any]) -> Dict[str, Any]:
    name = _clean(params.get("name"))
    instructions = _clean(params.get("instructions"))
    if not name or not instructions:
        return {"ok": False, "local_only": True, "error": "name_and_instructions_required"}
    data = _load()
    digest = hashlib.sha256((name + "\n" + instructions).encode("utf-8")).hexdigest()
    for item in data["skills"]:
        if item.get("digest") == digest:
            return {"ok": True, "local_only": True, "deduplicated": True, "skill": item}
    item = {"id": "skill_" + uuid.uuid4().hex[:12], "name": name, "instructions": instructions, "triggers": params.get("triggers", []), "state": "candidate", "digest": digest, "created_at": _now(), "updated_at": _now()}
    data["skills"].append(item)
    _event(data, "bridge.create_skill", {"skill_id": item["id"]})
    _save(data)
    return {"ok": True, "local_only": True, "deduplicated": False, "skill": item}


def record_feedback(params: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    item = {"id": "feedback_" + uuid.uuid4().hex[:12], "operation": _clean(params.get("operation")), "outcome": _clean(params.get("outcome")), "correction": _clean(params.get("correction")), "evidence": params.get("evidence", []), "at": _now()}
    if not item["operation"] or not item["outcome"]:
        return {"ok": False, "local_only": True, "error": "operation_and_outcome_required"}
    data["feedback"].append(item)
    _event(data, "bridge.record_feedback", {"feedback_id": item["id"]})
    _save(data)
    return {"ok": True, "local_only": True, "feedback": item}


def list_knowledge(params: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    kind = _clean(params.get("kind"))
    query = _clean(params.get("query")).lower()
    groups = {"lessons": data.get("lessons", []), "skills": data.get("skills", []), "feedback": data.get("feedback", [])}
    if kind and kind in groups:
        groups = {kind: groups[kind]}
    if query:
        groups = {key: [item for item in items if query in json.dumps(item, ensure_ascii=False).lower()] for key, items in groups.items()}
    return {"ok": True, "local_only": True, "knowledge": groups, "counts": {key: len(items) for key, items in groups.items()}}


def ingest_file(params: Dict[str, Any]) -> Dict[str, Any]:
    path = _safe_path(params.get("path"))
    if not path.is_file() or path.stat().st_size > 5_000_000:
        return {"ok": False, "local_only": True, "error": "valid_local_file_required"}
    text = path.read_text(encoding="utf-8", errors="replace")
    return record_lesson({"title": _clean(params.get("title")) or path.name, "lesson": text, "source": str(path), "tags": params.get("tags", [])})


def snapshot(params: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    target = SNAPSHOT_DIR / ("snapshot_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + ".json")
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "local_only": True, "path": str(target), "counts": {key: len(data.get(key, [])) for key in ("lessons", "skills", "feedback", "events")}}


def status(params: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    return {"ok": True, "local_only": True, "version": data.get("version", 1), "path": str(STORE_PATH), "counts": {key: len(data.get(key, [])) for key in ("lessons", "skills", "feedback", "events")}, "external_sync": False, "model_weights_exported": False}


_HANDLERS = {"bridge.record_lesson": record_lesson, "bridge.promote_lesson": promote_lesson, "bridge.create_skill": create_skill, "bridge.record_feedback": record_feedback, "bridge.list_knowledge": list_knowledge, "bridge.ingest_file": ingest_file, "bridge.snapshot": snapshot, "bridge.status": status}


def dispatch(operation: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        handler = _HANDLERS.get(operation)
        if handler is None:
            return {"ok": False, "local_only": True, "error": "unknown_operation", "operation": operation}
        return handler(params or {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ok": False, "local_only": True, "error": type(exc).__name__, "detail": str(exc)}
