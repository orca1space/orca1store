from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(os.environ.get("HERMES_ROOT", Path(__file__).resolve().parents[1]))
STATE_FILE = ROOT / "data" / "mentor_mode.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default() -> Dict[str, Any]:
    return {
        "version": 1,
        "mode": "disconnected",
        "mentor": None,
        "session_id": None,
        "capture_enabled": False,
        "promotion_enabled": False,
        "started_at": None,
        "last_transition_at": _now(),
        "observations": 0,
        "promotions": 0,
        "external_connection": False,
        "local_only": True,
        "internal_reasoning_excluded": True,
    }


def _load() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return _default()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        base = _default()
        base.update(data)
        return base
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return _default()


def _save(data: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def status(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _load()
    data["local_only"] = True
    data["external_connection"] = False
    data["independent_execution_available"] = True
    data["learning_capture_active"] = bool(data.get("capture_enabled")) and data.get("mode") == "connected"
    return {"ok": True, **data}


def connect(params: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    if data.get("mode") == "connected":
        return {"ok": True, "already_connected": True, **status({})}
    mentor = str(params.get("mentor", "Manus")).strip() or "Manus"
    data.update({
        "mode": "connected",
        "mentor": mentor,
        "session_id": "mentor_" + uuid.uuid4().hex[:12],
        "capture_enabled": True,
        "promotion_enabled": bool(params.get("promotion_enabled", True)),
        "started_at": _now(),
        "last_transition_at": _now(),
        "external_connection": False,
        "local_only": True,
        "internal_reasoning_excluded": True,
    })
    _save(data)
    return {"ok": True, "connected": True, "note": "Mentor mode is active locally; only declared execution data is captured.", **status({})}


def disconnect(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _load()
    data.update({
        "mode": "disconnected",
        "capture_enabled": False,
        "promotion_enabled": False,
        "last_transition_at": _now(),
        "external_connection": False,
        "local_only": True,
    })
    _save(data)
    return {"ok": True, "disconnected": True, "note": "Independent local execution remains available; new mentor observations are no longer captured.", **status({})}


def should_capture() -> bool:
    data = _load()
    return data.get("mode") == "connected" and bool(data.get("capture_enabled"))


def mark_observation() -> Dict[str, Any]:
    data = _load()
    data["observations"] = int(data.get("observations", 0)) + 1
    _save(data)
    return data


def mark_promotion() -> Dict[str, Any]:
    data = _load()
    data["promotions"] = int(data.get("promotions", 0)) + 1
    _save(data)
    return data


_HANDLERS = {
    "mentor.status": status,
    "mentor.connect": connect,
    "mentor.disconnect": disconnect,
}


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
