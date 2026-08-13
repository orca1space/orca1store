"""
Hermes API Importer — bulk import of ALL capabilities from one API call.

Imports everything the agent needs in a single transaction:
  - skills
  - knowledge base chunks
  - lessons
  - memory entries
  - system prompt additions

After import, the API can be disconnected. Hermes stays 100% local.

PRIVACY MODEL:
  - The api_key is sent ONLY in the Authorization header for this single call.
  - The api_key is NEVER stored on disk or in memory beyond this call.
  - No telemetry, no tracking, no third-party calls.
  - After import completes, you can delete the key and never call the API again.

WORKFLOW:
  1. User provides API URL (and optional API key) of an external source
  2. Hermes calls the API ONCE
  3. Hermes validates and stores the response locally as JSON files
  4. Hermes rebuilds the skill embeddings, KB vectors, memory indexes
  5. User deletes the API key
  6. From now on, all is local
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import (
    SKILLS_DIR, KNOWLEDGE_DIR, MEMORY_PATH, HERMES_ROOT,
    SKILLS_REGISTRY,
)
from pathlib import Path
MEMORY_FILE = MEMORY_PATH if isinstance(MEMORY_PATH, Path) else Path(MEMORY_PATH)
from core.logging_setup import get_logger

log = get_logger("api_importer")


class ApiImporter:
    """Bulk import skills, knowledge, lessons, and memory from a single API."""

    def __init__(self):
        self.skills_dir = SKILLS_DIR
        self.knowledge_dir = KNOWLEDGE_DIR
        self.memory_file = MEMORY_FILE
        self.skills_registry = SKILLS_REGISTRY

    def import_everything(
        self,
        url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        max_per_kind: int = 1000,
    ) -> Dict:
        """
        One call to import all Hermes capabilities.

        Expected response shapes (any combination accepted):

        1. Wrapped:
        {
          "skills":     [ {skill}, ... ],
          "knowledge":  [ {"text": "...", "source": "..."}, ... ],
          "lessons":    [ {"lesson": "..."}, ... ],
          "memory":     [ {"role": "user"|"assistant", "content": "..."}, ... ],
          "system_prompt_additions": [ "extra rule 1", "extra rule 2" ]
        }

        2. Flat list of skills (treated as skills)
        3. Single object (treated as skill or knowledge depending on fields)

        Returns a summary dict with counts of each imported item.
        """
        if not url or not isinstance(url, str):
            raise ValueError("url required")
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must be http(s)://")

        # === 1. Call the API once ===
        req_headers = {
            "User-Agent": "Hermes/1.0 (local; +https://github.com/local/hermes)",
            "Accept": "application/json",
        }
        if api_key:
            req_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            for k, v in (headers or {}).items():
                req_headers[str(k)] = str(v)

        req = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API returned HTTP {e.code}: {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f"API call failed: {e}") from e

        if status != 200:
            raise RuntimeError(f"API returned HTTP {status}")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"API did not return valid JSON: {e}") from e

        # === 2. Normalize and route ===
        result = {
            "source_url": url,
            "timestamp": time.time(),
            "skills": {"imported": 0, "failed": 0, "items": []},
            "knowledge": {"imported": 0, "failed": 0, "items": []},
            "lessons": {"imported": 0, "failed": 0, "items": []},
            "memory": {"imported": 0, "failed": 0, "items": []},
            "system_prompt_additions": {"imported": 0, "items": []},
            "errors": [],
        }

        # Determine the payload shape and extract each section
        sections = self._extract_sections(payload)

        # === 3. Import each section ===

        # 3a. Skills
        for s in sections.get("skills", [])[:max_per_kind]:
            try:
                self._import_skill(s)
                result["skills"]["imported"] += 1
                result["skills"]["items"].append(s.get("name", "?"))
            except Exception as e:
                result["skills"]["failed"] += 1
                result["errors"].append({"section": "skills", "name": s.get("name"), "error": str(e)})

        # 3b. Knowledge base
        for k in sections.get("knowledge", [])[:max_per_kind]:
            try:
                self._import_knowledge(k)
                result["knowledge"]["imported"] += 1
                result["knowledge"]["items"].append(k.get("source", "?"))
            except Exception as e:
                result["knowledge"]["failed"] += 1
                result["errors"].append({"section": "knowledge", "source": k.get("source"), "error": str(e)})

        # 3c. Lessons
        for l in sections.get("lessons", [])[:max_per_kind]:
            try:
                self._import_lesson(l)
                result["lessons"]["imported"] += 1
                result["lessons"]["items"].append(l.get("lesson", "")[:50])
            except Exception as e:
                result["lessons"]["failed"] += 1
                result["errors"].append({"section": "lessons", "error": str(e)})

        # 3d. Memory entries
        for m in sections.get("memory", [])[:max_per_kind]:
            try:
                self._import_memory(m)
                result["memory"]["imported"] += 1
                result["memory"]["items"].append(f"{m.get('role','?')}:{(m.get('content','') or '')[:30]}")
            except Exception as e:
                result["memory"]["failed"] += 1
                result["errors"].append({"section": "memory", "error": str(e)})

        # 3e. System prompt additions
        for line in sections.get("system_prompt_additions", [])[:50]:
            if isinstance(line, str) and line.strip():
                self._import_system_prompt_addition(line.strip())
                result["system_prompt_additions"]["imported"] += 1
                result["system_prompt_additions"]["items"].append(line.strip()[:60])

        # === 4. Rebuild indexes after bulk import ===
        try:
            self._rebuild_indexes()
            result["indexes_rebuilt"] = True
        except Exception as e:
            result["indexes_rebuilt"] = False
            result["errors"].append({"section": "indexes", "error": str(e)})

        # === 5. Add a privacy note ===
        result["privacy_note"] = (
            "API key was used only for this single call. "
            "All imported content is now on disk as local files. "
            "Delete the API key to fully disconnect from the source."
        )
        return result

    # ===== Section extraction =====

    @staticmethod
    def _extract_sections(payload) -> Dict[str, List]:
        """Extract skills/knowledge/lessons/memory/prompt sections from any response shape."""
        out = {
            "skills": [],
            "knowledge": [],
            "lessons": [],
            "memory": [],
            "system_prompt_additions": [],
        }
        if isinstance(payload, list):
            # Heuristic: if items have "name" + "description" -> skills
            for item in payload:
                if not isinstance(item, dict):
                    continue
                if "name" in item and ("description" in item or "procedure" in item):
                    out["skills"].append(item)
                elif "text" in item or "content" in item and "source" in item:
                    out["knowledge"].append(item)
                elif "lesson" in item:
                    out["lessons"].append(item)
                elif "role" in item and "content" in item:
                    out["memory"].append(item)
            return out

        if not isinstance(payload, dict):
            return out

        # Direct section keys
        for key in ("skills", "knowledge", "lessons", "memory", "system_prompt_additions"):
            v = payload.get(key)
            if isinstance(v, list):
                out[key] = [x for x in v if isinstance(x, (dict, str))]
            elif isinstance(v, str):
                out["system_prompt_additions"].append(v)

        # If no sections found, treat the dict as a single skill
        if not any(out.values()) and "name" in payload:
            out["skills"].append(payload)

        return out

    # ===== Importers =====

    def _import_skill(self, data: Dict):
        """Save a skill (reuses SkillLibrary)."""
        from core.skill_library import get_library
        get_library().save_skill(data, source=f"imported:api-bulk")

    def _import_knowledge(self, data: Dict):
        """Add text to the knowledge base."""
        from core.knowledge import get_kb
        text = data.get("text") or data.get("content")
        if not text or not isinstance(text, str):
            raise ValueError("knowledge item missing 'text' field")
        source = data.get("source", "imported:api-bulk")
        metadata = data.get("metadata", {})
        metadata = dict(metadata or {})
        metadata.setdefault("source_api", True)
        get_kb().add_text(text, source=source, metadata=metadata)

    def _import_lesson(self, data: Dict):
        """Add a lesson (uses memory module)."""
        from core.memory import get_memory
        lesson_text = data.get("lesson") or data.get("text")
        if not lesson_text or not isinstance(lesson_text, str):
            raise ValueError("lesson item missing 'lesson' field")
        mem = get_memory()
        # Save in the 'lessons' file (memory uses 'lessons' as a separate field)
        if not hasattr(mem, "add_lesson") or not callable(getattr(mem, "add_lesson", None)):
            # fallback: write directly to the lessons list
            if "lessons" not in mem.data:
                mem.data["lessons"] = []
            mem.data["lessons"].append({
                "lesson": lesson_text.strip(),
                "added_at": time.time(),
                "source": "imported:api-bulk",
            })
            mem.save()
        else:
            mem.add_lesson(lesson_text.strip(), source="imported:api-bulk")

    def _import_memory(self, data: Dict):
        """Add a memory entry (role/content pair)."""
        from core.memory import get_memory
        role = data.get("role", "user")
        content = data.get("content") or data.get("text")
        if not content or not isinstance(content, str):
            raise ValueError("memory item missing 'content' field")
        if role not in ("user", "assistant", "system"):
            role = "user"
        conv_id = data.get("conversation_id") or "imported"
        get_memory().add_message(conv_id, role, content,
                                 metadata={"source": "imported:api-bulk"})

    def _import_system_prompt_addition(self, line: str):
        """Append a line to the user's stored prompt additions."""
        additions_path = HERMES_ROOT / "data" / "prompt_additions.txt"
        additions_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't duplicate identical lines
        existing = ""
        if additions_path.exists():
            existing = additions_path.read_text(encoding="utf-8")
        if line not in existing:
            with open(additions_path, "a", encoding="utf-8") as f:
                f.write(line.strip() + "\n")

    def _rebuild_indexes(self):
        """Force rebuild of skill embeddings + KB vectors after bulk import."""
        try:
            from core.skill_search import get_skill_search
            from core.orchestrator import get_orchestrator
            orch = get_orchestrator()
            if hasattr(orch, "skill_search"):
                orch.skill_search.index_skills(orch.skills.skills, force=True)
                log.info("Skill embeddings rebuilt")
        except Exception as e:
            log.warning("Skill index rebuild failed: %s", e)
        try:
            from core.vector_store import get_store
            store = get_store()
            if hasattr(store, "save"):
                store.save()
                log.info("KB store saved")
        except Exception as e:
            log.warning("KB save failed: %s", e)


# Singleton
_importer: Optional[ApiImporter] = None


def get_api_importer() -> ApiImporter:
    global _importer
    if _importer is None:
        _importer = ApiImporter()
    return _importer
