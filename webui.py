"""
OrcaMax Code â€” Application Window
Local-only server + native window launcher.
Opens as a standalone app window (no browser chrome).

Features:
- Streaming chat (SSE) for instant first-token
- Persistent session restore on startup
- Pre-warmed model (no cold start on first request)
- File uploads + cache + quick replies
"""
import http.server
import socketserver
import json
import os
import sys
import threading
import subprocess
import time
import re
from pathlib import Path

# Make Hermes core importable
HERMES_ROOT = Path(__file__).parent
sys.path.insert(0, str(HERMES_ROOT))

PORT = 7777
WEBUI_DIR = HERMES_ROOT / "webui"


def parse_multipart(content_type: str, body: bytes):
    """
    Minimal multipart/form-data parser.
    Returns list of (name, filename, content_bytes) tuples.
    No external dependencies. Python 3.14 compatible (cgi was removed).
    """
    m = re.search(r'boundary=([^;\s]+)', content_type, re.IGNORECASE)
    if not m:
        raise ValueError("No boundary in Content-Type")
    boundary = m.group(1).strip('"')
    delimiter = b"--" + boundary.encode("ascii")
    parts = body.split(delimiter)
    results = []
    for part in parts:
        if part in (b"", b"--\r\n", b"--"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if b"\r\n\r\n" in part:
            header_block, file_body = part.split(b"\r\n\r\n", 1)
        else:
            continue
        header_text = header_block.decode("utf-8", errors="replace")
        cd_match = re.search(
            r'Content-Disposition:\s*form-data;\s*name="([^"]+)"(?:;\s*filename="([^"]*)")?',
            header_text, re.IGNORECASE
        )
        if not cd_match:
            continue
        name = cd_match.group(1)
        filename = cd_match.group(2) or ""
        results.append((name, filename, file_body))
    return results


# === Input validation ===
MAX_MESSAGE_LEN = 32_000       # 32K chars max per chat message
MAX_TAB_TITLE_LEN = 200
MAX_JSON_BODY = 5_000_000      # 5 MB max JSON request body
MAX_QUERY_LEN = 2_000
MAX_FILENAME_LEN = 255

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\.]{1,64}$")


class ValidationError(ValueError):
    """Raised when user input fails validation. Returns HTTP 400."""
    pass


def _get_json_body(self, max_bytes: int = MAX_JSON_BODY) -> dict:
    """Read and parse the request body as JSON with a size limit."""
    length = int(self.headers.get("Content-Length", 0))
    if length <= 0:
        return {}
    if length > max_bytes:
        raise ValidationError(f"Request body too large: {length} > {max_bytes}")
    raw = self.rfile.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}")


def _validate_message(msg) -> str:
    """Validate and normalize a chat message."""
    if not isinstance(msg, str):
        raise ValidationError("message must be a string")
    msg = msg.strip()
    if not msg:
        raise ValidationError("Empty message")
    if len(msg) > MAX_MESSAGE_LEN:
        raise ValidationError(f"Message too long ({len(msg)} > {MAX_MESSAGE_LEN})")
    return msg


def _validate_tab_id(tab_id) -> str:
    """Validate tab id (alphanumeric + dashes)."""
    if not isinstance(tab_id, str) or not _SAFE_NAME.match(tab_id or ""):
        raise ValidationError("Invalid tab_id")
    return tab_id


def _validate_skill_name(name) -> str:
    if not isinstance(name, str) or not _SAFE_NAME.match(name or ""):
        raise ValidationError("Invalid skill name")
    return name


def _validate_path(path_str, must_exist: bool = False) -> str:
    """Validate a filesystem path. Reject anything outside HERMES_ROOT.

    Prevents path traversal in the loaders/ingest endpoint.
    """
    if not isinstance(path_str, str) or not path_str:
        raise ValidationError("Path must be a non-empty string")
    if len(path_str) > 4096:
        raise ValidationError("Path too long")
    # Normalize and check it's under HERMES_ROOT
    p = Path(path_str).resolve()
    root = HERMES_ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise ValidationError(f"Path outside HERMES_ROOT: {path_str}")
    if must_exist and not p.exists():
        raise ValidationError(f"Path does not exist: {path_str}")
    return str(p)


def _validate_int(value, default: int, min_v: int, max_v: int, name: str) -> int:
    if value is None:
        return default
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be an integer")
    if v < min_v or v > max_v:
        raise ValidationError(f"{name} out of range [{min_v}, {max_v}]")
    return v


class OrcaMaxHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves the UI and proxies chat to Hermes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBUI_DIR), **kwargs)

    def log_message(self, format, *args):
        pass

    def _path(self) -> str:
        """Strip query string from self.path."""
        return self.path.split("?", 1)[0]

    def do_POST(self):
        p = self._path()
        if p == "/api/chat":
            self.handle_chat()
        elif p == "/api/chat/stream":
            self.handle_chat_stream()
        elif p == "/api/status":
            self.handle_status()
        elif p == "/api/upload":
            self.handle_upload()
        elif p == "/api/cache/clear":
            self.handle_cache_clear()
        elif p == "/api/session/clear":
            self.handle_session_clear()
        elif p == "/api/tabs":
            self.handle_tabs_create()
        elif p == "/api/tabs/switch":
            self.handle_tabs_switch()
        elif p == "/api/tabs/close":
            self.handle_tabs_close()
        elif p == "/api/tabs/rename":
            self.handle_tabs_rename()
        elif p == "/api/skills/search":
            self.handle_skills_search()
        elif p == "/api/skills/import":
            self.handle_skills_import()
        elif p == "/api/skills/import_api":
            self.handle_skills_import_api()
        elif p == "/api/import/full":
            self.handle_import_full()
        elif p == "/api/training/start":
            self.handle_training_start()
        elif p == "/api/training/pause":
            self.handle_training_pause()
        elif p == "/api/training/resume":
            self.handle_training_resume()
        elif p == "/api/training/stop":
            self.handle_training_stop()
        elif p == "/api/training/status":
            self.handle_training_status()
        elif p == "/api/training/add":
            self.handle_training_add()
        elif p == "/api/skills/learn":
            self.handle_skills_learn()
        elif p == "/api/skills/reindex":
            self.handle_skills_reindex()
        elif p == "/api/graph/run":
            self.handle_graph_run()
        elif p == "/api/checkpoint/save":
            self.handle_checkpoint_save()
        elif p == "/api/checkpoint/list":
            self.handle_checkpoint_list()
        elif p == "/api/checkpoint/fork":
            self.handle_checkpoint_fork()
        elif p == "/api/timetravel/history":
            self.handle_timetravel_history()
        elif p == "/api/timetravel/diff":
            self.handle_timetravel_diff()
        elif p == "/api/memory_store":
            self.handle_memory_store()
        elif p == "/api/multi_agent/run":
            self.handle_multi_agent_run()
        elif p == "/api/hitl/list":
            self.handle_hitl_list()
        elif p == "/api/hitl/approve":
            self.handle_hitl_approve()
        elif p == "/api/hitl/reject":
            self.handle_hitl_reject()
        elif p == "/api/hybrid_search":
            self.handle_hybrid_search()
        elif p == "/api/loaders/ingest":
            self.handle_loaders_ingest()
        elif p == "/api/agent/exec":
            self.handle_agent_exec()
        elif p == "/api/agent/ops":
            self.handle_agent_ops()
        elif p == "/api/agent/token":
            self.handle_agent_token()
        elif p.startswith("/api/skills/") and p.count("/") == 3:
            skill_name = p.rsplit("/", 1)[-1]
            self.handle_skill_get(skill_name)
        else:
            self.send_error(404)

    def do_GET(self):
        p = self._path()
        if p == "/api/status":
            self.handle_status()
        elif p == "/api/uploads":
            self.handle_uploads_list()
        elif p == "/api/cache":
            self.handle_cache_stats()
        elif p == "/api/session":
            self.handle_session_get()
        elif p == "/api/tabs":
            self.handle_tabs_list()
        elif p.startswith("/api/tabs/") and p.count("/") == 3:
            tab_id = p.rsplit("/", 1)[-1]
            self.handle_tabs_get(tab_id)
        elif p == "/api/skills":
            self.handle_skills_list()
        elif p.startswith("/api/skills/") and p.count("/") == 3:
            skill_name = p.rsplit("/", 1)[-1]
            self.handle_skill_get(skill_name)
        elif p == "/api/training/status":
            self.handle_training_status()
        elif p == "/api/agent/ops":
            self.handle_agent_ops()
        elif p == "/api/agent/token":
            self.handle_agent_token()
        else:
            super().do_GET()

    # --- Handlers ---

    def handle_chat(self):
        try:
            data = _get_json_body(self)
            user_message = _validate_message(data.get("message"))
            use_cache = bool(data.get("use_cache", True))
            tab_id = data.get("tab_id")
            if tab_id is not None:
                tab_id = _validate_tab_id(tab_id)
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            response = orch.chat(user_message, stream=False, use_cache=use_cache, tab_id=tab_id)
            self.json_response({"response": response, "tab_id": orch.session.get_active_tab_id()})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_chat_stream(self):
        """SSE streaming endpoint. Sends 'data: <json>\\n\\n' lines."""
        try:
            data = _get_json_body(self)
            user_message = _validate_message(data.get("message"))
            use_cache = bool(data.get("use_cache", True))
            tab_id = data.get("tab_id")
            if tab_id is not None:
                tab_id = _validate_tab_id(tab_id)

            # SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "http://localhost:7777")
            self.end_headers()

            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            for kind, content in orch.chat_stream(user_message, use_cache=use_cache, tab_id=tab_id):
                payload = json.dumps(
                    {"kind": kind, "content": content, "tab_id": orch.session.get_active_tab_id()},
                    ensure_ascii=False,
                )
                # SSE format
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
            return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            try:
                err = json.dumps({"kind": "error", "content": str(e)}, ensure_ascii=False)
                self.wfile.write(f"data: {err}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception:
                # Last resort: log and stop. Don't crash the SSE stream.
                import logging
                logging.exception("SSE stream error")

    def handle_status(self):
        try:
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            status = orch.status()
            self.json_response(status)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_upload(self):
        try:
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                self.json_response({"error": "Expected multipart/form-data"}, 400)
                return

            # Pre-check size before reading the whole body
            length = int(self.headers.get("Content-Length", 0))
            from core.file_uploads import MAX_FILE_SIZE
            if length > MAX_FILE_SIZE:
                self.json_response(
                    {"error": f"File too large: {length} > {MAX_FILE_SIZE} bytes"}, 413
                )
                return

            body = self.rfile.read(length)
            parts = parse_multipart(ctype, body)
            file_part = next((p for p in parts if p[0] == "file"), None)
            if not file_part:
                self.json_response({"error": "Missing 'file' field"}, 400)
                return
            _, filename, content = file_part
            if not filename or not content:
                self.json_response({"error": "No file selected"}, 400)
                return
            if len(filename) > MAX_FILENAME_LEN:
                self.json_response(
                    {"error": f"Filename too long ({len(filename)} > {MAX_FILENAME_LEN})"}, 400
                )
                return
            # Reject obvious path-traversal attempts in filename
            if "/" in filename or "\\" in filename or ".." in filename:
                self.json_response({"error": "Invalid filename"}, 400)
                return

            from core.file_uploads import get_uploader
            uploader = get_uploader()
            result = uploader.ingest(filename, content)
            self.json_response(result)
        except ValueError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_uploads_list(self):
        try:
            from core.file_uploads import get_uploader
            uploader = get_uploader()
            self.json_response({"uploads": uploader.list_uploads()})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_cache_stats(self):
        try:
            from core.cache import get_cache
            self.json_response(get_cache().stats())
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_cache_clear(self):
        try:
            from core.cache import get_cache
            get_cache().clear()
            self.json_response({"ok": True})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_session_get(self):
        try:
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            state = orch.restore_last_session()
            self.json_response(state)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_session_clear(self):
        try:
            from core.session_state import get_session
            get_session().clear()
            self.json_response({"ok": True})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === TAB HANDLERS ===

    def handle_tabs_list(self):
        try:
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            tabs = orch.list_tabs()
            active = orch.session.get_active_tab_id()
            self.json_response({"tabs": tabs, "active_tab_id": active})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_tabs_get(self, tab_id):
        try:
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            tab = orch.get_tab(tab_id)
            if tab is None:
                self.json_response({"error": "Tab not found"}, 404)
                return
            self.json_response({"tab": tab})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_tabs_create(self):
        try:
            data = _get_json_body(self, max_bytes=10_000)
            title = data.get("title")
            if title is not None:
                if not isinstance(title, str):
                    raise ValidationError("title must be a string")
                title = title.strip()[:MAX_TAB_TITLE_LEN] or None
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            new_tab = orch.create_tab(title)
            self.json_response({"tab": new_tab})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_tabs_switch(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            data = json.loads(body) if body else {}
            tab_id = data.get("tab_id")
            if not tab_id:
                self.json_response({"error": "Missing tab_id"}, 400)
                return
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            tab = orch.switch_tab(tab_id)
            if tab is None:
                self.json_response({"error": "Tab not found"}, 404)
                return
            self.json_response({"tab": {"id": tab["id"], "title": tab.get("title", "")}})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_tabs_close(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            data = json.loads(body) if body else {}
            tab_id = data.get("tab_id")
            if not tab_id:
                self.json_response({"error": "Missing tab_id"}, 400)
                return
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            ok = orch.close_tab(tab_id)
            if not ok:
                self.json_response({"error": "Tab not found"}, 404)
                return
            self.json_response({
                "ok": True,
                "active_tab_id": orch.session.get_active_tab_id(),
            })
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_tabs_rename(self):
        try:
            data = _get_json_body(self, max_bytes=10_000)
            tab_id = _validate_tab_id(data.get("tab_id", ""))
            title = data.get("title", "")
            if not isinstance(title, str):
                raise ValidationError("title must be a string")
            title = title.strip()[:MAX_TAB_TITLE_LEN]
            from core.session_state import get_session
            ok = get_session().rename_tab(tab_id, title)
            if not ok:
                self.json_response({"error": "Tab not found"}, 404)
                return
            self.json_response({"ok": True})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === SKILL HANDLERS ===

    def handle_skills_list(self):
        try:
            from core.orchestrator import get_orchestrator
            o = get_orchestrator()
            skills = o.skills.list_all()
            self.json_response({"skills": skills, "count": len(skills)})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_skill_get(self, skill_name):
        try:
            from core.orchestrator import get_orchestrator
            o = get_orchestrator()
            skill = o.skills.get(skill_name)
            if not skill:
                self.json_response({"error": "Skill not found"}, 404)
                return
            self.json_response({"skill": skill.data})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_skills_search(self):
        """Semantic search over skills."""
        try:
            data = _get_json_body(self, max_bytes=20_000)
            query = (data.get("query") or "").strip()
            if not query:
                raise ValidationError("Missing 'query'")
            if len(query) > MAX_QUERY_LEN:
                raise ValidationError(f"Query too long ({len(query)} > {MAX_QUERY_LEN})")
            top_k = _validate_int(data.get("top_k", 5), default=5, min_v=1, max_v=50, name="top_k")
            try:
                min_score = float(data.get("min_score", 0.30))
            except (TypeError, ValueError):
                raise ValidationError("min_score must be a number")
            if not (0.0 <= min_score <= 1.0):
                raise ValidationError("min_score must be in [0, 1]")
            from core.orchestrator import get_orchestrator
            o = get_orchestrator()
            results = o.skill_search.search(query, top_k=top_k, min_score=min_score)
            out = [
                {
                    "name": name,
                    "score": round(score, 3),
                    "description": skill.description,
                    "triggers": skill.triggers,
                    "enabled": skill.enabled,
                }
                for name, score, skill in results
            ]
            self.json_response({"results": out, "query": query, "count": len(out)})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_skills_import(self):
        """Import a skill from text (JSON or pasted code block)."""
        try:
            data = _get_json_body(self, max_bytes=200_000)
            text = data.get("text", "")
            if not isinstance(text, str) or not text:
                raise ValidationError("Missing 'text'")
            if len(text) > 100_000:
                raise ValidationError(f"Skill text too long ({len(text)} > 100K)")
            from core.skill_library import get_library
            result = get_library().import_from_text(text)
            self.json_response({"imported": result})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except ValueError as ve:
            self.json_response({"error": str(ve)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_skills_import_api(self):
        """
        Import skills from an external API endpoint.

        After this call returns, the skills are saved as local JSON files.
        The API key is used ONLY for this single call â€” it is never persisted.
        You can delete the API key afterwards and Hermes stays 100% local.
        """
        try:
            data = _get_json_body(self, max_bytes=10_000)
            url = data.get("url", "")
            if not isinstance(url, str) or not url:
                raise ValidationError("Missing 'url'")
            if not url.startswith(("http://", "https://")):
                raise ValidationError("url must start with http:// or https://")
            if len(url) > 2048:
                raise ValidationError("URL too long (max 2048)")
            api_key = data.get("api_key")
            if api_key is not None and not isinstance(api_key, str):
                raise ValidationError("api_key must be a string")
            if api_key and len(api_key) > 1024:
                raise ValidationError("api_key too long (max 1024)")
            headers = data.get("headers")
            if headers is not None and not isinstance(headers, dict):
                raise ValidationError("headers must be a dict")
            timeout = _validate_int(data.get("timeout", 30), default=30, min_v=5, max_v=120, name="timeout")
            max_skills = _validate_int(data.get("max_skills", 200), default=200, min_v=1, max_v=2000, name="max_skills")

            from core.skill_library import get_library
            # Wipe the api_key from local frame right after the call
            result = get_library().import_from_api(
                url=url, api_key=api_key,
                headers=headers, timeout=timeout,
                max_skills=max_skills,
            )
            # Note: we DO NOT echo the api_key in the response
            result.pop("api_key", None)
            self.json_response(result)
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except (ValueError, RuntimeError) as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_import_full(self):
        """
        Bulk import ALL Hermes capabilities from a single API call.

        Imports in one go:
          - skills
          - knowledge base chunks
          - lessons
          - memory entries
          - system prompt additions

        After this call, you can delete the API key. Hermes stays 100% local.
        """
        try:
            data = _get_json_body(self, max_bytes=20_000)
            url = data.get("url", "")
            if not isinstance(url, str) or not url:
                raise ValidationError("Missing 'url'")
            if not url.startswith(("http://", "https://")):
                raise ValidationError("url must start with http:// or https://")
            if len(url) > 2048:
                raise ValidationError("URL too long (max 2048)")
            api_key = data.get("api_key")
            if api_key is not None and not isinstance(api_key, str):
                raise ValidationError("api_key must be a string")
            if api_key and len(api_key) > 1024:
                raise ValidationError("api_key too long (max 1024)")
            headers = data.get("headers")
            if headers is not None and not isinstance(headers, dict):
                raise ValidationError("headers must be a dict")
            timeout = _validate_int(data.get("timeout", 60), default=60, min_v=5, max_v=300, name="timeout")
            max_per_kind = _validate_int(data.get("max_per_kind", 500), default=500, min_v=1, max_v=10000, name="max_per_kind")

            from core.api_importer import get_api_importer
            result = get_api_importer().import_everything(
                url=url, api_key=api_key, headers=headers,
                timeout=timeout, max_per_kind=max_per_kind,
            )
            # Note: NEVER echo api_key in response
            result.pop("api_key", None)
            self.json_response(result)
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except (ValueError, RuntimeError) as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_skills_learn(self):
        """Learn a new skill from a user description."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            data = _get_json_body(self, max_bytes=200_000)
            name = _validate_skill_name((data.get("name") or "").strip())
            description = (data.get("description") or "").strip()
            procedure = (data.get("procedure") or "").strip()
            if not description or not procedure:
                raise ValidationError("description and procedure required")
            if len(description) > 2000 or len(procedure) > 50_000:
                raise ValidationError("description/procedure too long")
            tab_id = data.get("tab_id")
            if tab_id is not None:
                tab_id = _validate_tab_id(tab_id)
            from core.skill_library import get_library
            if tab_id:
                result = get_library().learn_from_interaction(tab_id, name, description)
            else:
                triggers = data.get("triggers", [])
                if not isinstance(triggers, list):
                    raise ValidationError("triggers must be a list")
                triggers = [str(t)[:100] for t in triggers[:20]]
                result = get_library().learn_skill(name, description, procedure,
                                                  triggers=triggers)
            self.json_response({"learned": result})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_skills_reindex(self):
        """Rebuild the skill embeddings cache."""
        try:
            from core.orchestrator import get_orchestrator
            o = get_orchestrator()
            n = o.skill_search.index_skills(o.skills.skills, force=True)
            self.json_response({"reindexed": n})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === GRAPH ===

    def handle_graph_run(self):
        try:
            from core.graph import get_graph_executor, GraphState
            from core.checkpoint import get_checkpoint_manager
            data = _get_json_body(self, max_bytes=500_000)
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            conditional = data.get("conditional", [])
            # Structural validation
            if not isinstance(nodes, list) or len(nodes) > 200:
                raise ValidationError("nodes must be a list of <= 200 items")
            if not isinstance(edges, list) or len(edges) > 500:
                raise ValidationError("edges must be a list of <= 500 items")
            if not isinstance(conditional, list) or len(conditional) > 200:
                raise ValidationError("conditional must be a list of <= 200 items")
            # Ensure each node has required fields
            node_ids = set()
            for n in nodes:
                if not isinstance(n, dict) or "id" not in n:
                    raise ValidationError("each node needs an 'id'")
                if not isinstance(n["id"], str) or not _SAFE_NAME.match(n["id"]):
                    raise ValidationError(f"invalid node id: {n.get('id')!r}")
                if n["id"] in node_ids:
                    raise ValidationError(f"duplicate node id: {n['id']}")
                node_ids.add(n["id"])
            spec = {
                "name": str(data.get("name", "graph"))[:64],
                "nodes": nodes,
                "edges": edges,
                "conditional": conditional,
                "entry": data.get("entry"),
                "finish": data.get("finish", []),
            }
            cp = None
            if data.get("execution_id"):
                if not isinstance(data["execution_id"], str):
                    raise ValidationError("execution_id must be a string")
                cp = get_checkpoint_manager()
            executor = get_graph_executor(spec, checkpoint_manager=cp)
            execution_id = data.get("execution_id") or executor.execution_id
            initial = GraphState(values=dict(data.get("initial_state", {})))
            final = executor.run(initial)
            self.json_response({
                "execution_id": execution_id,
                "final_state": final.values if hasattr(final, "values") else dict(final),
                "trace": executor.get_node_results(),
                "steps": len(executor.node_results),
                "interrupted_at": executor._interrupted,
            })
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === CHECKPOINT HANDLERS ===

    def handle_checkpoint_save(self):
        try:
            from core.checkpoint import get_checkpoint_manager
            data = _get_json_body(self, max_bytes=5_000_000)
            eid = str(data.get("execution_id", "manual"))[:128]
            step = _validate_int(data.get("step", 0), default=0, min_v=0, max_v=10_000, name="step")
            node = str(data.get("node", "manual"))[:128]
            state = data.get("state", {})
            if not isinstance(state, dict):
                raise ValidationError("state must be a dict")
            cm = get_checkpoint_manager()
            cp = cm.save(
                execution_id=eid,
                step=step,
                node_name=node,
                state=state,
                metadata=data.get("metadata"),
            )
            self.json_response({"checkpoint_id": cp.id})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_checkpoint_list(self):
        try:
            from core.checkpoint import get_checkpoint_manager
            from urllib.parse import parse_qs
            cm = get_checkpoint_manager()
            qs = parse_qs(self.path.split("?", 1)[-1] if "?" in self.path else "")
            eid = (qs.get("eid", [""])[0]) if qs else ""
            history = cm.get_history(eid) if eid else []
            self.json_response({"checkpoints": [
                {"id": cp.id, "step": cp.step, "node": cp.node_name, "created_at": cp.created_at}
                for cp in history
            ]})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_checkpoint_fork(self):
        try:
            from core.time_travel import get_time_travel
            data = _get_json_body(self, max_bytes=20_000)
            eid = str(data.get("execution_id") or "")[:128]
            if not eid:
                raise ValidationError("execution_id required")
            step = _validate_int(data.get("step", 0), default=0, min_v=0, max_v=10_000, name="step")
            new_id = data.get("new_id")
            if new_id is not None:
                new_id = str(new_id)[:128]
            tt = get_time_travel()
            out = tt.fork(
                source_execution_id=eid,
                fork_step=step,
                new_execution_id=new_id,
            )
            self.json_response({"new_execution_id": out})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === TIME TRAVEL ===

    def handle_timetravel_history(self):
        try:
            from core.time_travel import get_time_travel
            from urllib.parse import parse_qs
            tt = get_time_travel()
            qs = parse_qs(self.path.split("?", 1)[-1] if "?" in self.path else "")
            eid = (qs.get("eid", [""])[0]) if qs else ""
            history = tt.list_history(eid) if eid else []
            self.json_response({"history": history})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_timetravel_diff(self):
        try:
            from core.time_travel import get_time_travel
            data = _get_json_body(self, max_bytes=20_000)
            eid = str(data.get("execution_id") or "")[:128]
            if not eid:
                raise ValidationError("execution_id required")
            step1 = _validate_int(data.get("step1", 0), default=0, min_v=0, max_v=10_000, name="step1")
            step2 = _validate_int(data.get("step2", 0), default=0, min_v=0, max_v=10_000, name="step2")
            tt = get_time_travel()
            diff = tt.diff(
                execution_id=eid,
                step1=step1,
                step2=step2,
            )
            self.json_response({"diff": diff})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === MEMORY STORE ===

    def handle_memory_store(self):
        try:
            from core.memory_store import get_memory_store
            data = _get_json_body(self, max_bytes=200_000)
            action = data.get("action", "get")
            if not isinstance(action, str) or action not in (
                "put", "get", "search", "list_namespaces", "list_keys", "delete", "stats",
            ):
                raise ValidationError(f"Unknown action: {action!r}")
            ms = get_memory_store()
            ns = str(data.get("namespace", "default"))[:64]
            key = str(data.get("key", ""))[:256]
            if action == "put":
                ttl = data.get("ttl")
                if ttl is not None and (not isinstance(ttl, int) or ttl < 0 or ttl > 365 * 24 * 3600):
                    raise ValidationError("ttl must be a non-negative integer in seconds")
                value = data.get("value")
                if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                    raise ValidationError("value must be JSON-serializable")
                eid = ms.put(ns, key, value,
                             ttl_seconds=ttl,
                             metadata=data.get("metadata"))
                self.json_response({"id": eid})
            elif action == "get":
                val = ms.get(ns, key)
                self.json_response({"value": val})
            elif action == "search":
                q = str(data.get("query", ""))[:MAX_QUERY_LEN]
                limit = _validate_int(data.get("limit", 10), default=10, min_v=1, max_v=100, name="limit")
                results = ms.search(ns, q, limit=limit)
                self.json_response({"results": results})
            elif action == "list_namespaces":
                self.json_response({"namespaces": ms.list_namespaces()})
            elif action == "list_keys":
                self.json_response({"keys": ms.list_keys(ns)})
            elif action == "delete":
                ok = ms.delete(ns, key)
                self.json_response({"deleted": ok})
            elif action == "stats":
                self.json_response({"stats": ms.stats()})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === MULTI-AGENT ===

    def handle_multi_agent_run(self):
        try:
            from core.multi_agent import get_multi_agent
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            data = json.loads(body) if body else {}
            ma = get_multi_agent()
            self.json_response({
                "agents": ma.list_agents(),
                "stats": ma.stats(),
            })
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === HUMAN-IN-THE-LOOP ===

    def handle_hitl_list(self):
        try:
            from core.human_in_loop import get_hitl
            hitl = get_hitl()
            self.json_response({"pending": hitl.list_pending()})
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_hitl_approve(self):
        try:
            from core.human_in_loop import get_hitl
            data = _get_json_body(self, max_bytes=10_000)
            iid = str(data.get("interrupt_id") or "")
            if not iid or len(iid) > 128:
                raise ValidationError("interrupt_id required")
            hitl = get_hitl()
            ok = hitl.approve(iid)
            self.json_response({"approved": ok})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_hitl_reject(self):
        try:
            from core.human_in_loop import get_hitl
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            data = _get_json_body(self, max_bytes=10_000)
            iid = str(data.get("interrupt_id") or "")
            if not iid or len(iid) > 128:
                raise ValidationError("interrupt_id required")
            reason = str(data.get("reason") or "")[:500]
            hitl = get_hitl()
            ok = hitl.reject(iid, reason)
            self.json_response({"rejected": ok})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === HYBRID SEARCH ===

    def handle_hybrid_search(self):
        try:
            from core.orchestrator import get_orchestrator
            data = _get_json_body(self, max_bytes=20_000)
            query = str(data.get("query") or "").strip()
            if not query:
                raise ValidationError("query required")
            if len(query) > MAX_QUERY_LEN:
                raise ValidationError(f"query too long ({len(query)} > {MAX_QUERY_LEN})")
            top_k = _validate_int(data.get("top_k", 5), default=5, min_v=1, max_v=50, name="top_k")
            o = get_orchestrator()
            results = o.hybrid_search.search(
                query=query,
                top_k=top_k,
            )
            self.json_response({"results": results, "count": len(results)})
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === LOADERS ===

    def handle_loaders_ingest(self):
        try:
            from core.loaders import ingest_to_kb
            data = _get_json_body(self, max_bytes=20_000)
            # Validate path is under HERMES_ROOT (prevents traversal)
            source = _validate_path(data.get("source", ""), must_exist=True)
            loader_type = data.get("loader_type")
            if loader_type is not None and loader_type not in (
                "filesystem", "zip", "git", "archive",
            ):
                raise ValidationError(f"Unknown loader_type: {loader_type!r}")
            result = ingest_to_kb(
                source=source,
                loader_type=loader_type,
                recursive=bool(data.get("recursive", True)),
            )
            self.json_response(result)
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    # === AGENT API (unified command interface) ===

    def _check_agent_auth(self, body, allow_token_in_body=True):
        """Returns (ok, error_dict). By default OrcaMax has no auth â€” caller
        is always accepted. To enforce, set $HERMES_AGENT_TOKEN and
        $HERMES_AGENT_AUTH=1 in the environment before launching."""
        from core.agent_api import check_auth, extract_token
        token = extract_token(self.headers, body if allow_token_in_body else None)
        if not check_auth(token):
            return False, {"error": "unauthorized",
                           "hint": "auth is enforced â€” set HERMES_AGENT_TOKEN and HERMES_AGENT_AUTH=1 to enable"}
        return True, None

    def handle_agent_exec(self):
        """POST /api/agent/exec â€” single op or batch."""
        try:
            try:
                data = _get_json_body(self, max_bytes=2_000_000)
            except ValidationError:
                data = {}
            if not isinstance(data, dict):
                self.json_response({"ok": False, "error": "body must be JSON object"}, 400)
                return
            ok, err = self._check_agent_auth(data, allow_token_in_body=True)
            if not ok:
                self.json_response({"ok": False, **err}, 401)
                return
            from core.agent_api import execute
            response = execute(data)
            # Always 200 â€” failures are inside payload {ok:false, error:...}
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
        except Exception as e:
            self.json_response({"ok": False, "error": f"server error: {e}"}, 500)

    def handle_agent_ops(self):
        """GET /api/agent/ops â€” list all registered ops (no auth needed for discovery)."""
        try:
            from core.agent_api import _OPS, API_VERSION
            ops = sorted(_OPS.keys())
            self.json_response({"ok": True, "api_version": API_VERSION, "count": len(ops), "ops": ops})
        except Exception as e:
            self.json_response({"ok": False, "error": str(e)}, 500)

    def handle_agent_token(self):
        """GET /api/agent/token â€” disabled by default (OrcaMax is auth-free).
        Returns 410 Gone with a hint, unless auth is explicitly enforced."""
        try:
            from core.agent_api import get_token, _auth_enforced
            if not _auth_enforced():
                self.json_response({
                    "ok": False,
                    "error": "auth disabled â€” OrcaMax is local-only, no token needed",
                    "hint": "set HERMES_AGENT_TOKEN and HERMES_AGENT_AUTH=1 to opt in to token auth"
                }, 410)
                return
            self.json_response({"ok": True, "token": get_token()})
        except Exception as e:
            self.json_response({"ok": False, "error": str(e)}, 500)

    # === TRAINING DAEMON ===

    def handle_training_start(self):
        try:
            data = _get_json_body(self, max_bytes=20_000) if self.headers.get("Content-Length", "0") != "0" else {}
            budget_hours = float(data.get("budget_hours", 10.0))
            budget_seconds = budget_hours * 3600.0
            if budget_hours < 0.1 or budget_hours > 168:
                raise ValidationError("budget_hours must be 0.1-168")
            resume = bool(data.get("resume", True))
            sources = data.get("sources")
            from core.training_daemon import get_training_daemon
            result = get_training_daemon().start(
                budget_seconds=budget_seconds,
                sources=sources,
                resume=resume,
            )
            self.json_response(result)
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_training_pause(self):
        try:
            from core.training_daemon import get_training_daemon
            self.json_response(get_training_daemon().pause())
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_training_resume(self):
        try:
            from core.training_daemon import get_training_daemon
            self.json_response(get_training_daemon().resume())
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_training_stop(self):
        try:
            from core.training_daemon import get_training_daemon
            self.json_response(get_training_daemon().stop())
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_training_status(self):
        try:
            from core.training_daemon import get_training_daemon
            self.json_response(get_training_daemon().status())
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def handle_training_add(self):
        try:
            data = _get_json_body(self, max_bytes=10_000)
            sources = data.get("sources", [])
            if not isinstance(sources, list) or not sources:
                raise ValidationError("sources must be a non-empty list")
            for s in sources:
                if not isinstance(s, dict) or "type" not in s:
                    raise ValidationError("each source needs 'type'")
            from core.training_daemon import get_training_daemon
            self.json_response(get_training_daemon().add_sources(sources))
        except ValidationError as e:
            self.json_response({"error": str(e)}, 400)
        except Exception as e:
            self.json_response({"error": str(e)}, 500)

    def json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "http://localhost:7777")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def find_browser():
    candidates = [
        ("msedge", "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"),
        ("msedge", "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe"),
        ("chrome", "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"),
        ("chrome", "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"),
    ]
    for exe_name, path in candidates:
        if os.path.exists(path):
            return path
    return None


def open_app_window(port, delay=3):
    time.sleep(delay)
    url = f"http://localhost:{port}/"
    browser_path = find_browser()
    if browser_path:
        subprocess.Popen(
            [browser_path, f"--app={url}", "--window-size=1200,800", "--window-position=200,100"],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        import webbrowser
        webbrowser.open(url)


def main():
    print(f"=" * 50)
    print(f"  OrcaMax Code")
    print(f"  Local: http://localhost:{PORT}/")
    print(f"=" * 50)
    print()
    print("  Pre-warming model (this happens once per launch)...")
    print()

    # Pre-warm model in background while window opens
    def warmup():
        try:
            from core.llm import get_llm
            get_llm().pre_warm()
            from core.orchestrator import get_orchestrator
            state = get_orchestrator().restore_last_session()
            if state.get("messages"):
                print(f"  [Session] Restored {len(state['messages'])} messages from last session.")
        except Exception as e:
            print(f"  [Warmup] Warning: {e}")

    threading.Thread(target=warmup, daemon=True).start()
    # Open app window after 2s
    threading.Thread(target=open_app_window, args=(PORT, 2), daemon=True).start()
    # Start server
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), OrcaMaxHandler) as httpd:
        httpd.daemon_threads = True
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[OrcaMax] Stopped.")


if __name__ == "__main__":
    main()

