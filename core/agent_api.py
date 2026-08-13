"""
OrcaMax Code — Agent API
Unified command-based interface for any external AI agent to fully control
OrcaMax (Hermes): skills, KB, lessons, memory, training, sessions, cache,
config, chat, prompt, imports, graph, system.

NO AUTH by default — OrcaMax is local-only and independent. The server binds
to 127.0.0.1:7777, so access is already restricted to the local machine.

To OPT-IN to token auth (e.g. if you bind the server to a public interface):
  set $HERMES_AGENT_TOKEN  → required bearer token
  set $HERMES_AGENT_AUTH=1 → turn on enforcement even on localhost
  Otherwise: any caller on localhost is accepted.

Usage:
  POST /api/agent/exec
  {
    "op": "skills.create",
    "params": {"name": "my_skill", "content": "...", "tags": [...]}
  }

  Or batch:
  {
    "ops": [
      {"op": "skills.list", "params": {}},
      {"op": "kb.search", "params": {"query": "python"}}
    ]
  }

All responses: {"ok": true, "result": ...}  or  {"ok": false, "error": "..."}
"""
import os
import json
import time
import secrets
import hashlib
import traceback
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Local imports (lazy inside ops so import is cheap)
HERMES_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = HERMES_ROOT / "data"
API_VERSION = "1.0"

# ----------------------------------------------------------------------------
# Auth (off by default — OrcaMax is local-only, independence first)
# ----------------------------------------------------------------------------

# Auth is enforced only when both conditions are true:
#   1. $HERMES_AGENT_TOKEN is set (the expected bearer token)
#   2. $HERMES_AGENT_AUTH=1  (explicit opt-in to enforcement)
# Otherwise every caller is accepted. The server still binds to 127.0.0.1.
_TOKEN: Optional[str] = None
_TOKEN_LOCK = threading.Lock()


def _auth_enforced() -> bool:
    """Auth is enforced only when explicitly opted in via env vars."""
    return bool(os.environ.get("HERMES_AGENT_TOKEN", "").strip()) and \
           os.environ.get("HERMES_AGENT_AUTH", "").strip() in ("1", "true", "yes", "on")


def _load_token() -> str:
    """Return the current bearer token (only valid if auth is enforced)."""
    global _TOKEN
    with _TOKEN_LOCK:
        if _TOKEN:
            return _TOKEN
        env = os.environ.get("HERMES_AGENT_TOKEN", "").strip()
        if env:
            _TOKEN = env
            return _TOKEN
        return ""


def check_auth(provided: Optional[str]) -> bool:
    """No-op when auth is not enforced. Returns True.
    When enforced, performs constant-time compare against expected token.
    """
    if not _auth_enforced():
        return True
    if not provided:
        return False
    expected = _load_token()
    if not expected:
        return False
    try:
        return secrets.compare_digest(str(provided), expected)
    except Exception:
        return False


def get_token() -> str:
    """Return the active token (or empty string if auth is not enforced)."""
    return _load_token() if _auth_enforced() else ""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _ok(result: Any = None) -> Dict[str, Any]:
    return {"ok": True, "result": result, "api_version": API_VERSION}


def _err(msg: str, code: str = "error", **extra) -> Dict[str, Any]:
    out = {"ok": False, "error": str(msg), "code": code, "api_version": API_VERSION}
    out.update(extra)
    return out


def _require(cond: bool, msg: str, code: str = "bad_request"):
    if not cond:
        raise ValueError(f"{code}:{msg}")


def _safe_name(name: str, max_len: int = 64) -> str:
    import re
    if not name or not isinstance(name, str):
        raise ValueError("bad_request: name must be a non-empty string")
    if not re.match(r"^[A-Za-z0-9_\-\.]{1," + str(max_len) + "}$", name):
        raise ValueError(
            f"bad_request: name must match ^[A-Za-z0-9_\\-.]{1,{max_len}}$"
        )
    return name


def _jsonable(obj: Any) -> Any:
    """Best-effort convert non-JSON objects to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return obj.hex()
    return str(obj)


# ----------------------------------------------------------------------------
# Lazy module accessors
# ----------------------------------------------------------------------------

def _skills():
    from core import skills as _m
    return _m


def _skill_library():
    from core import skill_library as _m
    return _m


def _skill_search():
    from core import skill_search as _m
    return _m


def _knowledge():
    from core import knowledge as _m
    return _m


def _memory():
    from core import memory as _m
    return _m


def _memory_store():
    from core import memory_store as _m
    return _m


def _session_state():
    from core import session_state as _m
    return _m


def _cache():
    from core import cache as _m
    return _m


def _training():
    from core import training_daemon as _m
    return _m


def _config():
    from core import config as _m
    return _m


def _llm():
    from core import llm as _m
    return _m


def _api_importer():
    from core import api_importer as _m
    return _m


def _graph():
    from core import graph as _m
    return _m


def _hybrid_search():
    from core import hybrid_search as _m
    return _m


def _loaders():
    from core import loaders as _m
    return _m


def _orchestrator():
    from core import orchestrator as _m
    return _m


def _checkpoint():
    from core import checkpoint as _m
    return _m


def _time_travel():
    from core import time_travel as _m
    return _m


def _human_in_loop():
    from core import human_in_loop as _m
    return _m


def _multi_agent():
    from core import multi_agent as _m
    return _m


# ----------------------------------------------------------------------------
# Op registry
# ----------------------------------------------------------------------------

_OPS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def op(name: str):
    def deco(fn: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], Any]:
        _OPS[name] = fn
        return fn
    return deco


# -------- System --------

@op("system.ping")
def _op_ping(params: Dict[str, Any]) -> Any:
    return {"pong": True, "ts": time.time()}


@op("system.info")
def _op_info(params: Dict[str, Any]) -> Any:
    info = {
        "api_version": API_VERSION,
        "hermes_root": str(HERMES_ROOT),
        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.exists(),
        "uptime_hint": time.time(),
        "python": os.sys.version.split()[0] if os.sys else "unknown",
    }
    # Try to report model path
    try:
        cfg = _config()
        info["model_path"] = str(getattr(cfg, "LLM_MODEL_PATH", ""))
        info["llm_loaded"] = bool(getattr(_llm()._llm_instance, "model", None) or getattr(_llm()._llm_instance, "_model", None))
    except Exception:
        pass
    return info


@op("system.shutdown")
def _op_shutdown(params: Dict[str, Any]) -> Any:
    import threading, os as _os
    def _kill():
        time.sleep(0.5)
        _os._exit(0)
    threading.Thread(target=_kill, daemon=True).start()
    return {"shutting_down": True}


@op("system.list_ops")
def _op_list_ops(params: Dict[str, Any]) -> Any:
    return sorted(_OPS.keys())


# -------- Skills --------

SKILLS_DIR = HERMES_ROOT / "skills"


# ----------------------------------------------------------------------------
# Instance accessors (modules expose factories like get_kb(), get_library()…)
# ----------------------------------------------------------------------------

_INSTANCE_CACHE: Dict[str, Any] = {}


def _inst(cache_key: str, factory):
    """Lazy singleton: call factory() once, cache result, return."""
    if cache_key in _INSTANCE_CACHE:
        return _INSTANCE_CACHE[cache_key]
    inst = factory()
    _INSTANCE_CACHE[cache_key] = inst
    return inst


def _kb_inst():
    return _inst("kb", lambda: _knowledge().get_kb())


def _library_inst():
    return _inst("library", lambda: _skill_library().get_library())


def _skill_search_inst():
    return _inst("skill_search", lambda: _skill_search().get_skill_search())


def _memory_inst():
    return _inst("memory", lambda: _memory().get_memory())


def _memory_store_inst():
    return _inst("memory_store", lambda: _memory_store().get_memory_store())


def _session_state_inst():
    return _inst("session_state", lambda: _session_state().get_session())


def _cache_inst():
    return _inst("cache", lambda: _cache().get_cache())


def _training_inst():
    return _inst("training", lambda: _training().get_training_daemon())


def _orchestrator_inst():
    return _inst("orchestrator", lambda: _orchestrator().get_orchestrator() if hasattr(_orchestrator(), "get_orchestrator") else _orchestrator())


def _hybrid_search_inst():
    return _inst("hybrid_search", lambda: _hybrid_search().get_hybrid_search())


def _loaders_inst():
    return _inst("loaders", lambda: _loaders())


def _human_in_loop_inst():
    return _inst("hitl", lambda: _human_in_loop())


def _multi_agent_inst():
    return _inst("multi_agent", lambda: _multi_agent())


def _checkpoint_inst():
    return _inst("checkpoint", lambda: _checkpoint())


def _time_travel_inst():
    return _inst("time_travel", lambda: _time_travel())


def _graph_inst():
    return _inst("graph", lambda: _graph())


def _llm_inst():
    return _inst("llm", lambda: _llm())


def _api_importer_inst():
    return _inst("api_importer", lambda: _api_importer())


def _list_skill_files() -> List[Path]:
    if not SKILLS_DIR.exists():
        return []
    return sorted(p for p in SKILLS_DIR.glob("*.json") if p.name != "imported_registry.json")


def _read_skill_file(name: str) -> Optional[Dict[str, Any]]:
    safe = _safe_name(name)
    path = SKILLS_DIR / f"{safe}.json"
    if not path.exists() or path.name == "imported_registry.json":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"read_error: cannot parse {path.name}: {e}")


def _write_skill_file(name: str, data: Dict[str, Any]) -> Path:
    safe = _safe_name(name)
    if safe == "imported_registry":
        raise ValueError("bad_request: reserved name 'imported_registry'")
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = SKILLS_DIR / f"{safe}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _delete_skill_file(name: str) -> bool:
    safe = _safe_name(name)
    path = SKILLS_DIR / f"{safe}.json"
    if not path.exists() or path.name == "imported_registry.json":
        return False
    path.unlink()
    return True


def _build_skill_payload(name: str, content: str, description: str = "",
                         tags: Optional[List[str]] = None,
                         triggers: Optional[List[str]] = None,
                         examples: Optional[List[Dict]] = None,
                         category: str = "general") -> Dict[str, Any]:
    """Build a skill dict matching the project's expected schema."""
    tags = tags or []
    triggers = triggers or []
    if not triggers and tags:
        triggers = list(tags)
    if not description:
        # Pull first non-empty line as description
        for line in content.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                description = line[:200]
                break
    payload = {
        "name": name,
        "description": description or f"Skill '{name}'",
        "category": category,
        "trigger_keywords": triggers,
        "tags": tags,
        "procedure": content,
    }
    if examples:
        payload["examples"] = examples
    return payload


@op("skills.list")
def _op_skills_list(params: Dict[str, Any]) -> Any:
    files = _list_skill_files()
    return _jsonable([p.stem for p in files])


@op("skills.get")
def _op_skills_get(params: Dict[str, Any]) -> Any:
    name = _safe_name(params.get("name", ""))
    data = _read_skill_file(name)
    if data is None:
        raise FileNotFoundError(f"not_found: skill '{name}' does not exist")
    return _jsonable(data)


@op("skills.create")
def _op_skills_create(params: Dict[str, Any]) -> Any:
    name = _safe_name(params.get("name", ""))
    content = params.get("content") or params.get("body") or params.get("procedure")
    _require(content and isinstance(content, str), "content/procedure required (string)", "bad_request")
    description = params.get("description") or params.get("desc") or ""
    tags = params.get("tags") or []
    triggers = params.get("triggers") or params.get("trigger_keywords")
    examples = params.get("examples")
    category = params.get("category") or "agent_api"

    if (SKILLS_DIR / f"{name}.json").exists():
        raise FileExistsError(f"conflict: skill '{name}' already exists (use skills.update to overwrite)")

    payload = _build_skill_payload(name, content, description, tags, triggers, examples, category)
    _write_skill_file(name, payload)
    # Try library registration (optional, best-effort)
    try:
        lib = _library_inst()
        if hasattr(lib, "save_skill"):
            lib.save_skill(data=payload, source="imported:agent_api")
    except Exception:
        pass
    # Background reindex
    try:
        if hasattr(_skill_search(), "reindex"):
            threading.Thread(target=_skill_search().reindex, daemon=True).start()
    except Exception:
        pass
    return _jsonable({"created": name, "path": str(SKILLS_DIR / f"{name}.json")})


@op("skills.update")
def _op_skills_update(params: Dict[str, Any]) -> Any:
    name = _safe_name(params.get("name", ""))
    content = params.get("content") or params.get("body") or params.get("procedure")
    _require(content and isinstance(content, str), "content/procedure required (string)", "bad_request")
    description = params.get("description")
    tags = params.get("tags")
    triggers = params.get("triggers") or params.get("trigger_keywords")
    examples = params.get("examples")
    category = params.get("category")

    existing = _read_skill_file(name) or {}
    if description is None:
        description = existing.get("description", "")
    if tags is None:
        tags = existing.get("tags", [])
    if triggers is None:
        triggers = existing.get("trigger_keywords", [])
    if examples is None:
        examples = existing.get("examples")
    if category is None:
        category = existing.get("category", "general")

    payload = _build_skill_payload(name, content, description, tags, triggers, examples, category)
    _write_skill_file(name, payload)
    try:
        if hasattr(_skill_search(), "reindex"):
            threading.Thread(target=_skill_search().reindex, daemon=True).start()
    except Exception:
        pass
    return _jsonable({"updated": name})


@op("skills.delete")
def _op_skills_delete(params: Dict[str, Any]) -> Any:
    name = _safe_name(params.get("name", ""))
    deleted = _delete_skill_file(name)
    if not deleted:
        raise FileNotFoundError(f"not_found: skill '{name}' does not exist")
    try:
        if hasattr(_skill_search(), "reindex"):
            threading.Thread(target=_skill_search().reindex, daemon=True).start()
    except Exception:
        pass
    return _jsonable({"deleted": name})


@op("skills.search")
def _op_skills_search(params: Dict[str, Any]) -> Any:
    query = params.get("query") or params.get("q") or ""
    _require(query, "query required", "bad_request")
    top_k = int(params.get("top_k", 5))
    search = _skill_search_inst()
    import inspect
    fn = getattr(search, "search", None) or getattr(search, "find", None)
    if fn is None:
        raise RuntimeError("not_supported: skill_search has no search method")
    try:
        sig = inspect.signature(fn)
        kwargs = {}
        if "top_k" in sig.parameters: kwargs["top_k"] = top_k
        if "k" in sig.parameters: kwargs["k"] = top_k
        if "limit" in sig.parameters: kwargs["limit"] = top_k
        res = fn(query, **kwargs)
    except TypeError:
        res = fn(query)
    return _jsonable(res)


@op("skills.learn")
def _op_skills_learn(params: Dict[str, Any]) -> Any:
    name = _safe_name(params.get("name", ""))
    description = params.get("description") or ""
    procedure = params.get("procedure") or params.get("content") or ""
    _require(procedure, "procedure required (string)", "bad_request")
    examples = params.get("examples")
    triggers = params.get("triggers") or params.get("trigger_keywords")
    lib = _library_inst()
    if not hasattr(lib, "learn_skill"):
        raise RuntimeError("not_supported: library has no learn_skill")
    result = lib.learn_skill(
        name=name,
        description=description,
        procedure=procedure,
        examples=examples,
        triggers=triggers,
    )
    try:
        if hasattr(_skill_search(), "reindex"):
            threading.Thread(target=_skill_search().reindex, daemon=True).start()
    except Exception:
        pass
    return _jsonable({"learned": True, "result": result})


@op("skills.reindex")
def _op_skills_reindex(params: Dict[str, Any]) -> Any:
    search = _skill_search_inst()
    if hasattr(search, "reindex"):
        result = search.reindex()
    elif hasattr(search, "rebuild"):
        result = search.rebuild()
    else:
        raise RuntimeError("not_supported: search has no reindex method")
    return _jsonable({"reindexed": True, "result": result})


@op("skills.import_api")
def _op_skills_import_api(params: Dict[str, Any]) -> Any:
    url = params.get("url")
    api_key = params.get("api_key") or params.get("token")
    _require(url, "url required", "bad_request")
    lib = _library_inst()
    if not hasattr(lib, "import_from_api"):
        raise RuntimeError("not_supported: import_from_api not available")
    return _jsonable(lib.import_from_api(url=url, api_key=api_key))


# -------- KB --------

@op("kb.search")
def _op_kb_search(params: Dict[str, Any]) -> Any:
    query = params.get("query") or params.get("q") or ""
    _require(query, "query required", "bad_request")
    top_k = int(params.get("top_k", 3))
    threshold = float(params.get("threshold", 0.35))
    kb = _kb_inst()
    if hasattr(kb, "search"):
        try:
            import inspect
            sig = inspect.signature(kb.search)
            kwargs = {"top_k": top_k}
            if "threshold" in sig.parameters: kwargs["threshold"] = threshold
            return _jsonable(kb.search(query, **kwargs))
        except TypeError:
            return _jsonable(kb.search(query, top_k))
    return _jsonable(kb.query(query, top_k=top_k))


@op("kb.add")
def _op_kb_add(params: Dict[str, Any]) -> Any:
    text = params.get("text") or params.get("content")
    _require(text and isinstance(text, str), "text required", "bad_request")
    source = params.get("source") or "agent_api"
    tags = params.get("tags") or []
    metadata = params.get("metadata")
    if metadata is None:
        metadata = {"tags": tags, "added_via": "agent_api"}
    elif isinstance(metadata, dict) and tags and "tags" not in metadata:
        metadata["tags"] = tags
    kb = _kb_inst()
    fn = getattr(kb, "add_text", None) or getattr(kb, "ingest", None) or getattr(kb, "add", None)
    if fn is None:
        raise RuntimeError("not_supported: kb has no add method")
    try:
        import inspect
        sig = inspect.signature(fn)
        kwargs = {}
        if "text" in sig.parameters: kwargs["text"] = text
        if "source" in sig.parameters: kwargs["source"] = source
        if "metadata" in sig.parameters: kwargs["metadata"] = metadata
        if "tags" in sig.parameters: kwargs["tags"] = tags
        if "content" in sig.parameters: kwargs["content"] = text
        result = fn(**kwargs)
    except Exception as e:
        raise RuntimeError(f"kb_add failed: {e}")
    return _jsonable({"added": True, "ids": _jsonable(result), "count": len(result) if hasattr(result, '__len__') else 0})


@op("kb.hybrid_search")
def _op_kb_hybrid(params: Dict[str, Any]) -> Any:
    query = params.get("query") or ""
    _require(query, "query required", "bad_request")
    top_k = int(params.get("top_k", 5))
    hs = _hybrid_search_inst()
    if hasattr(hs, "search"):
        import inspect
        try:
            sig = inspect.signature(hs.search)
            kwargs = {}
            if "top_k" in sig.parameters: kwargs["top_k"] = top_k
            if "k" in sig.parameters: kwargs["k"] = top_k
            return _jsonable(hs.search(query, **kwargs))
        except TypeError:
            return _jsonable(hs.search(query))
    raise RuntimeError("not_supported: hybrid_search has no search method")


@op("kb.stats")
def _op_kb_stats(params: Dict[str, Any]) -> Any:
    kb = _kb_inst()
    if hasattr(kb, "stats"):
        return _jsonable(kb.stats())
    if hasattr(kb, "count"):
        return _jsonable({"chunks": kb.count()})
    return _jsonable({"chunks": "unknown"})


# -------- Lessons --------

@op("lessons.list")
def _op_lessons_list(params: Dict[str, Any]) -> Any:
    kb = _kb_inst()
    if hasattr(kb, "list_lessons"):
        return _jsonable(kb.list_lessons())
    return _jsonable([])


@op("lessons.add")
def _op_lessons_add(params: Dict[str, Any]) -> Any:
    name = _safe_name(params.get("name", ""))
    content = params.get("content") or params.get("text")
    _require(content and isinstance(content, str), "content required", "bad_request")
    language = params.get("language") or "general"
    kb = _kb_inst()
    if not hasattr(kb, "add_lesson"):
        raise RuntimeError("not_supported: knowledge has no add_lesson")
    return _jsonable(kb.add_lesson(name=name, content=content, language=language))


@op("lessons.delete")
def _op_lessons_delete(params: Dict[str, Any]) -> Any:
    name = _safe_name(params.get("name", ""))
    kb = _kb_inst()
    if hasattr(kb, "delete_lesson"):
        return _jsonable(kb.delete_lesson(name))
    raise RuntimeError("not_supported: knowledge has no delete_lesson")


# -------- Memory (short-term conversation) --------

@op("memory.add_message")
def _op_memory_add(params: Dict[str, Any]) -> Any:
    conv_id = params.get("tab_id") or params.get("conv_id") or "default"
    role = params.get("role") or "user"
    content = params.get("content") or ""
    _require(content, "content required", "bad_request")
    mem = _memory_inst()
    if hasattr(mem, "add_message"):
        return _jsonable(mem.add_message(conv_id, role, content))
    raise RuntimeError("not_supported: memory has no add_message")


@op("memory.get")
def _op_memory_get(params: Dict[str, Any]) -> Any:
    conv_id = params.get("tab_id") or params.get("conv_id") or "default"
    mem = _memory_inst()
    if hasattr(mem, "get_conversation"):
        return _jsonable(mem.get_conversation(conv_id))
    if hasattr(mem, "get"):
        return _jsonable(mem.get(conv_id))
    raise RuntimeError("not_supported: memory has no get_conversation")


@op("memory.clear")
def _op_memory_clear(params: Dict[str, Any]) -> Any:
    conv_id = params.get("tab_id") or params.get("conv_id")
    mem = _memory_inst()
    if hasattr(mem, "clear"):
        return _jsonable(mem.clear(conv_id) if conv_id else mem.clear_all())
    raise RuntimeError("not_supported: memory has no clear")


# -------- Memory store (long-term, key-value) --------

@op("memory_store.list")
def _op_memstore_list(params: Dict[str, Any]) -> Any:
    ns = params.get("namespace") or "default"
    ms = _memory_store_inst()
    if hasattr(ms, "list_keys"):
        return _jsonable(ms.list_keys(ns))
    if hasattr(ms, "keys"):
        return _jsonable(ms.keys(ns))
    raise RuntimeError("not_supported: memory_store has no list method")


@op("memory_store.get")
def _op_memstore_get(params: Dict[str, Any]) -> Any:
    key = params.get("key")
    _require(key, "key required", "bad_request")
    ns = params.get("namespace") or "default"
    ms = _memory_store_inst()
    if hasattr(ms, "get"):
        return _jsonable({"value": ms.get(ns, key)})
    raise RuntimeError("not_supported: memory_store has no get")


@op("memory_store.set")
def _op_memstore_set(params: Dict[str, Any]) -> Any:
    key = params.get("key")
    value = params.get("value")
    _require(key, "key required", "bad_request")
    ns = params.get("namespace") or "default"
    ttl = params.get("ttl")
    metadata = params.get("metadata")
    ms = _memory_store_inst()
    fn = getattr(ms, "put", None) or getattr(ms, "set", None)
    if fn is None:
        raise RuntimeError("not_supported: memory_store has no put/set")
    # MemoryStore.put signature: (namespace, key, value, ttl_seconds=None, metadata=None)
    args = [ns, key, value]
    kwargs = {}
    if ttl is not None: kwargs["ttl_seconds"] = int(ttl)
    if metadata is not None: kwargs["metadata"] = metadata
    result = fn(*args, **kwargs)
    return _jsonable({"set": True, "id": result})


@op("memory_store.delete")
def _op_memstore_delete(params: Dict[str, Any]) -> Any:
    key = params.get("key")
    _require(key, "key required", "bad_request")
    ns = params.get("namespace") or "default"
    ms = _memory_store_inst()
    if hasattr(ms, "delete"):
        return _jsonable({"deleted": ms.delete(ns, key)})
    raise RuntimeError("not_supported: memory_store has no delete")


# -------- Training daemon --------

@op("training.status")
def _op_train_status(params: Dict[str, Any]) -> Any:
    t = _training_inst()
    if hasattr(t, "get_status"):
        return _jsonable(t.get_status())
    if hasattr(t, "status"):
        return _jsonable(t.status())
    return _jsonable({"running": False, "note": "no status method"})


@op("training.start")
def _op_train_start(params: Dict[str, Any]) -> Any:
    budget_hours = float(params.get("budget_hours", 10))
    sources = params.get("sources") or []
    t = _training_inst()
    if hasattr(t, "start"):
        return _jsonable(t.start(budget_seconds=budget_hours * 3600.0, sources=sources))
    raise RuntimeError("not_supported: training_daemon has no start method")


@op("training.pause")
def _op_train_pause(params: Dict[str, Any]) -> Any:
    t = _training_inst()
    if hasattr(t, "pause"):
        return _jsonable({"paused": t.pause()})
    raise RuntimeError("not_supported: training_daemon has no pause")


@op("training.resume")
def _op_train_resume(params: Dict[str, Any]) -> Any:
    t = _training_inst()
    if hasattr(t, "resume"):
        return _jsonable({"resumed": t.resume()})
    raise RuntimeError("not_supported: training_daemon has no resume")


@op("training.stop")
def _op_train_stop(params: Dict[str, Any]) -> Any:
    t = _training_inst()
    if hasattr(t, "stop"):
        return _jsonable({"stopped": t.stop()})
    raise RuntimeError("not_supported: training_daemon has no stop")


@op("training.add")
def _op_train_add(params: Dict[str, Any]) -> Any:
    task = params.get("task") or params
    t = _training_inst()
    if hasattr(t, "add_task"):
        return _jsonable(t.add_task(task))
    raise RuntimeError("not_supported: training_daemon has no add_task")


# -------- Session / Tabs --------

@op("session.tabs")
def _op_session_tabs(params: Dict[str, Any]) -> Any:
    ss = _session_state_inst()
    if hasattr(ss, "list_tabs"):
        return _jsonable(ss.list_tabs())
    if hasattr(ss, "tabs"):
        return _jsonable(ss.tabs())
    raise RuntimeError("not_supported: session_state has no list_tabs")


@op("session.active")
def _op_session_active(params: Dict[str, Any]) -> Any:
    ss = _session_state_inst()
    if hasattr(ss, "get_active"):
        return _jsonable(ss.get_active())
    return _jsonable({})


@op("session.switch")
def _op_session_switch(params: Dict[str, Any]) -> Any:
    tab_id = _safe_name(params.get("tab_id", ""))
    ss = _session_state_inst()
    if hasattr(ss, "switch_tab"):
        return _jsonable(ss.switch_tab(tab_id))
    raise RuntimeError("not_supported: session_state has no switch_tab")


@op("session.create")
def _op_session_create(params: Dict[str, Any]) -> Any:
    name = params.get("name") or "New Tab"
    ss = _session_state_inst()
    if hasattr(ss, "create_tab"):
        return _jsonable(ss.create_tab(name=name))
    raise RuntimeError("not_supported: session_state has no create_tab")


@op("session.close")
def _op_session_close(params: Dict[str, Any]) -> Any:
    tab_id = _safe_name(params.get("tab_id", ""))
    ss = _session_state_inst()
    if hasattr(ss, "close_tab"):
        return _jsonable(ss.close_tab(tab_id))
    raise RuntimeError("not_supported: session_state has no close_tab")


@op("session.clear")
def _op_session_clear(params: Dict[str, Any]) -> Any:
    tab_id = params.get("tab_id")
    ss = _session_state_inst()
    if tab_id:
        if hasattr(ss, "clear_tab"):
            return _jsonable(ss.clear_tab(tab_id))
    if hasattr(ss, "clear_all"):
        return _jsonable(ss.clear_all())
    raise RuntimeError("not_supported: session_state has no clear methods")


# -------- Cache --------

@op("cache.stats")
def _op_cache_stats(params: Dict[str, Any]) -> Any:
    c = _cache_inst()
    if hasattr(c, "stats"):
        return _jsonable(c.stats())
    return _jsonable({"size": "unknown"})


@op("cache.clear")
def _op_cache_clear(params: Dict[str, Any]) -> Any:
    c = _cache_inst()
    if hasattr(c, "clear"):
        c.clear()
        return _jsonable({"cleared": True})
    raise RuntimeError("not_supported: cache has no clear")


@op("cache.peek")
def _op_cache_peek(params: Dict[str, Any]) -> Any:
    c = _cache_inst()
    if hasattr(c, "peek"):
        return _jsonable(c.peek(int(params.get("limit", 20))))
    return _jsonable([])


# -------- Config --------

@op("config.get")
def _op_config_get(params: Dict[str, Any]) -> Any:
    cfg = _config()
    key = params.get("key")
    if key:
        return _jsonable({key: getattr(cfg, key, None)})
    # dump all UPPERCASE
    return _jsonable({k: v for k, v in vars(cfg).items() if k.isupper()})


# -------- Chat / LLM --------

@op("chat.send")
def _op_chat_send(params: Dict[str, Any]) -> Any:
    message = params.get("message") or params.get("text") or ""
    _require(message, "message required", "bad_request")
    tab_id = params.get("tab_id") or "default"
    orch = _orchestrator_inst()
    if hasattr(orch, "handle_message"):
        out = orch.handle_message(message, tab_id=tab_id)
    elif hasattr(orch, "chat"):
        out = orch.chat(message, tab_id=tab_id)
    else:
        raise RuntimeError("not_supported: orchestrator has no chat method")
    return _jsonable({"response": out})


# -------- Prompt additions --------

_PROMPT_ADD_FILE = DATA_DIR / "prompt_additions.txt"


@op("prompt.additions.list")
def _op_prompt_add_list(params: Dict[str, Any]) -> Any:
    if not _PROMPT_ADD_FILE.exists():
        return []
    return _jsonable(_PROMPT_ADD_FILE.read_text(encoding="utf-8").splitlines())


@op("prompt.additions.add")
def _op_prompt_add_add(params: Dict[str, Any]) -> Any:
    text = params.get("text") or params.get("rule")
    _require(text and isinstance(text, str), "text required", "bad_request")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_PROMPT_ADD_FILE, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    return _jsonable({"added": True, "file": str(_PROMPT_ADD_FILE)})


@op("prompt.additions.clear")
def _op_prompt_add_clear(params: Dict[str, Any]) -> Any:
    if _PROMPT_ADD_FILE.exists():
        _PROMPT_ADD_FILE.write_text("", encoding="utf-8")
    return _jsonable({"cleared": True})


# -------- Graph --------

@op("graph.run")
def _op_graph_run(params: Dict[str, Any]) -> Any:
    spec = params.get("spec") or params
    g = _graph_inst()
    if hasattr(g, "build_graph_from_spec") and isinstance(spec, dict):
        executor = g.build_graph_from_spec(spec)
        initial = spec.get("initial_state", {})
    else:
        raise RuntimeError("not_supported: provide dict 'spec' for build_graph_from_spec")
    if hasattr(executor, "run"):
        result = executor.run(initial_state=initial)
    else:
        raise RuntimeError("not_supported: executor has no run method")
    return _jsonable(result)


# -------- API bulk import --------

@op("api.import_full")
def _op_api_import_full(params: Dict[str, Any]) -> Any:
    url = params.get("url")
    api_key = params.get("api_key") or params.get("token")
    _require(url, "url required", "bad_request")
    imp = _api_importer()
    if hasattr(imp, "import_full"):
        return _jsonable(imp.import_full(url=url, api_key=api_key))
    if hasattr(imp, "run"):
        return _jsonable(imp.run(url=url, api_key=api_key))
    raise RuntimeError("not_supported: api_importer has no import method")


@op("api.import_skills_only")
def _op_api_import_skills(params: Dict[str, Any]) -> Any:
    url = params.get("url")
    api_key = params.get("api_key") or params.get("token")
    _require(url, "url required", "bad_request")
    lib = _library_inst()
    if not hasattr(lib, "import_from_api"):
        raise RuntimeError("not_supported: import_from_api not available")
    return _jsonable(lib.import_from_api(url=url, api_key=api_key))


# -------- HITL --------

@op("hitl.list")
def _op_hitl_list(params: Dict[str, Any]) -> Any:
    h = _human_in_loop_inst()
    if hasattr(h, "list_pending"):
        return _jsonable(h.list_pending())
    if hasattr(h, "list_all"):
        return _jsonable(h.list_all())
    return _jsonable([])


@op("hitl.approve")
def _op_hitl_approve(params: Dict[str, Any]) -> Any:
    request_id = params.get("request_id") or params.get("id")
    _require(request_id, "request_id required", "bad_request")
    h = _human_in_loop_inst()
    if hasattr(h, "approve"):
        return _jsonable(h.approve(request_id))
    raise RuntimeError("not_supported: human_in_loop has no approve")


@op("hitl.reject")
def _op_hitl_reject(params: Dict[str, Any]) -> Any:
    request_id = params.get("request_id") or params.get("id")
    _require(request_id, "request_id required", "bad_request")
    reason = params.get("reason") or ""
    h = _human_in_loop_inst()
    if hasattr(h, "reject"):
        return _jsonable(h.reject(request_id, reason=reason))
    raise RuntimeError("not_supported: human_in_loop has no reject")


# -------- Checkpoints / time travel --------

@op("checkpoint.list")
def _op_ckpt_list(params: Dict[str, Any]) -> Any:
    c = _checkpoint_inst()
    if hasattr(c, "list_all"):
        return _jsonable(c.list_all())
    if hasattr(c, "list"):
        return _jsonable(c.list())
    return _jsonable([])


@op("time_travel.history")
def _op_tt_history(params: Dict[str, Any]) -> Any:
    t = _time_travel_inst()
    if hasattr(t, "list_history"):
        return _jsonable(t.list_history())
    return _jsonable([])


# -------- Loaders --------

@op("loaders.ingest")
def _op_loaders_ingest(params: Dict[str, Any]) -> Any:
    path = params.get("path")
    _require(path, "path required", "bad_request")
    loader_type = (params.get("type") or "filesystem").lower()
    L = _loaders_inst()
    if loader_type == "filesystem" and hasattr(L, "FileSystemLoader"):
        ld = L.FileSystemLoader(path)
    elif loader_type == "zip" and hasattr(L, "ZipLoader"):
        ld = L.ZipLoader(path)
    elif loader_type == "git" and hasattr(L, "GitLoader"):
        ld = L.GitLoader(path)
    elif loader_type == "archive" and hasattr(L, "ArchiveLoader"):
        ld = L.ArchiveLoader(path)
    else:
        raise RuntimeError(f"bad_request: unknown loader type '{loader_type}'")
    if hasattr(ld, "ingest"):
        return _jsonable(ld.ingest())
    raise RuntimeError("not_supported: loader has no ingest")


# -------- Multi-agent --------

@op("multi_agent.run")
def _op_multi_agent_run(params: Dict[str, Any]) -> Any:
    spec = params.get("spec") or params
    m = _multi_agent()
    if hasattr(m, "run"):
        return _jsonable(m.run(spec))
    raise RuntimeError("not_supported: multi_agent has no run")


# ----------------------------------------------------------------------------
from core.local_imports import register_local_import_ops
register_local_import_ops(op)

# Public dispatcher
# ----------------------------------------------------------------------------

def _record_live_execution(op_name: str, result: Any) -> None:
    if op_name.startswith("ledger.") or op_name in {"system.ping", "system.info"}:
        return
    try:
        from core.mentor_mode import should_capture, mark_observation
        if not should_capture():
            return
        mark_observation()
        from core.execution_ledger import record_execution
        record_execution({
            "task": f"Live operation: {op_name}",
            "objective": "Capture reusable execution pattern",
            "steps": [{"sequence": 1, "action": op_name, "result": "completed"}],
            "tools": [op_name],
            "outputs": {"result_type": type(result).__name__},
            "outcome": "completed",
            "approved": False,
        })
    except Exception:
        pass

def _execute_op(op_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if op_name not in _OPS:
        return _err(f"unknown op: {op_name}", code="unknown_op",
                    available=[k for k in sorted(_OPS.keys()) if k.startswith(op_name.split('.')[0] + '.')][:20])
    fn = _OPS[op_name]
    try:
        result = fn(params or {})
        safe_result = _jsonable(result)
        _record_live_execution(op_name, safe_result)
        return _ok(safe_result)
    except ValueError as e:
        msg = str(e)
        code = "bad_request"
        if ":" in msg:
            code = msg.split(":", 1)[0]
            msg = msg.split(":", 1)[1]
        return _err(msg, code=code)
    except FileNotFoundError as e:
        return _err(str(e), code="not_found")
    except PermissionError as e:
        return _err(str(e), code="forbidden")
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", code="exception",
                    trace=traceback.format_exc().splitlines()[-3:])


def execute(payload: Any) -> Dict[str, Any]:
    """Dispatch one payload. payload can be:
       {"op": "...", "params": {...}}
       {"op": "...", ...}                (params = whole body minus op)
       {"ops": [{"op": "...", "params": {...}}, ...]}
    """
    if not isinstance(payload, dict):
        return _err("payload must be a JSON object", code="bad_request")

    if "ops" in payload and isinstance(payload["ops"], list):
        results = []
        for item in payload["ops"]:
            if not isinstance(item, dict):
                results.append(_err("each op item must be an object", code="bad_request"))
                continue
            op_name = item.get("op") or item.get("operation") or ""
            params = item.get("params")
            if params is None:
                params = {k: v for k, v in item.items() if k not in ("op", "operation")}
            results.append(_execute_op(op_name, params))
        return {"ok": True, "batch": True, "count": len(results), "results": results,
                "api_version": API_VERSION}

    op_name = payload.get("op") or payload.get("operation") or ""
    params = payload.get("params")
    if params is None:
        params = {k: v for k, v in payload.items() if k not in ("op", "operation", "token")}
    return _execute_op(op_name, params)


def extract_token(headers, body) -> Optional[str]:
    """Pull bearer token from headers (Authorization) or body.token."""
    if headers:
        try:
            auth = headers.get("Authorization") or headers.get("authorization")
            if auth and isinstance(auth, str):
                a = auth.strip()
                if a.lower().startswith("bearer "):
                    return a.split(None, 1)[1].strip()
                if a:
                    return a
        except Exception:
            pass
    if isinstance(body, dict):
        tok = body.get("token") or body.get("auth")
        if tok:
            return str(tok)
    return None


# ----------------------------------------------------------------------------
# CLI self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("OrcaMax Agent API v" + API_VERSION)
    if _auth_enforced():
        print("Auth: ENFORCED — token required (Authorization: Bearer ...)")
        print("Token:", get_token() or "(set HERMES_AGENT_TOKEN)")
    else:
        print("Auth: DISABLED — OrcaMax is local-only, no token required.")
    print("Registered ops:", len(_OPS))
    for name in sorted(_OPS):
        print(" -", name)
