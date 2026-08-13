"""
Hermes Skill Library
Import / Save / Learn workflow for skills.
Pure local. No external services.

Operations:
- import_from_file: read a skill JSON from disk and register it
- import_from_text: parse a skill from a code block / pasted text
- save_skill: write a new or updated skill to disk
- learn_skill: capture a skill from a successful interaction (auto-extract procedure)
- export_skill: serialize a skill for sharing
- list_imported: track which skills were added via the library (vs built-in)
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import SKILLS_DIR, SKILLS_REGISTRY


# Track skills added through the library
IMPORTED_REGISTRY = SKILLS_DIR / "imported_registry.json"


class SkillLibrary:
    """Manages skill import/save/learn workflow."""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.dir = skills_dir
        self.imported = self._load_imported()

    def _load_imported(self) -> Dict:
        if not IMPORTED_REGISTRY.exists():
            return {"imported": []}
        try:
            with open(IMPORTED_REGISTRY, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"imported": []}

    def _save_imported(self):
        try:
            with open(IMPORTED_REGISTRY, "w", encoding="utf-8") as f:
                json.dump(self.imported, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def import_from_file(self, file_path: str) -> Dict:
        """Import a skill from a JSON file on disk."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Not found: {file_path}")
        if p.suffix.lower() != ".json":
            raise ValueError("Only .json skill files are supported")
        data = json.loads(p.read_text(encoding="utf-8"))
        return self.save_skill(data, source=f"imported:{p.name}")

    def import_from_text(self, text: str) -> Dict:
        """
        Parse a skill from pasted text. Accepts:
        - JSON object
        - JSON wrapped in ```json ... ``` code block
        """
        text = text.strip()
        # Strip code fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        # Try to find a JSON object
        if not text.startswith("{"):
            start = text.find("{")
            if start >= 0:
                # Find matching close
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            text = text[start:i+1]
                            break
        data = json.loads(text)
        return self.save_skill(data, source="imported:paste")

    def import_from_api(self, url: str, api_key: Optional[str] = None,
                        headers: Optional[Dict] = None,
                        timeout: int = 30,
                        max_skills: int = 500) -> Dict:
        """
        Import skills from an external API endpoint.

        PRIVACY:
        - The api_key is sent ONLY in the Authorization header for this call.
        - The api_key is NEVER stored on disk or in memory beyond this call.
        - After this call returns, the skills are saved as local JSON files.
        - No telemetry is sent, no analytics, no tracking.

        ACCEPTED RESPONSE FORMATS:
        - Single skill: {"name": "...", "description": "...", ...}
        - Skill list:   [{"name": "..."}, {"name": "..."}]
        - Wrapped:      {"skills": [...]}
        - Wrapped:      {"data": [...]}
        - HF dataset:   each row in "data" or "rows" with skill fields

        After import completes, you can delete the API key and never call
        the endpoint again. The imported skills persist locally.
        """
        import urllib.request
        import urllib.error

        if not url or not isinstance(url, str):
            raise ValueError("url required")
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must be http(s)://")

        # Build request with strict headers — no UA tracking
        req_headers = {
            "User-Agent": "Hermes/1.0 (local; +https://github.com/local/hermes)",
            "Accept": "application/json",
        }
        if api_key:
            # Use Authorization header. NEVER log the key.
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

        # Parse the response
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"API did not return valid JSON: {e}") from e

        # Normalize to list of skill dicts
        skills_list = self._extract_skills(payload)
        if not skills_list:
            raise ValueError("No skills found in API response")

        # Import each
        imported = []
        failed = []
        for skill_data in skills_list[:max_skills]:
            if not isinstance(skill_data, dict):
                failed.append({"error": "not a dict", "data": str(skill_data)[:80]})
                continue
            if "name" not in skill_data:
                failed.append({"error": "missing 'name'", "data": str(skill_data)[:80]})
                continue
            try:
                result = self.save_skill(skill_data, source=f"imported:api:{url[:60]}")
                imported.append(result)
            except Exception as e:
                failed.append({"name": skill_data.get("name"), "error": str(e)})

        return {
            "imported": imported,
            "failed": failed,
            "source_url": url,
            "total_in_response": len(skills_list),
            "imported_count": len(imported),
            "failed_count": len(failed),
            "note": "API key was used only for this single call. Delete it to fully disconnect.",
        }

    @staticmethod
    def _extract_skills(payload) -> List[Dict]:
        """Extract a list of skill dicts from any common API response shape."""
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if not isinstance(payload, dict):
            return []
        # Common wrapper keys
        for key in ("skills", "data", "rows", "items", "results", "records"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
            if isinstance(v, dict):
                # nested: {"skills": {"items": [...]}}
                for k2 in ("items", "rows", "data", "results"):
                    v2 = v.get(k2)
                    if isinstance(v2, list):
                        return [x for x in v2 if isinstance(x, dict)]
        # HF dataset style: {"features": [...], "rows": [[...], [...]]}
        if "features" in payload and "rows" in payload:
            features = payload["features"]
            if isinstance(features, list) and features:
                # Map feature names to indices
                names = []
                for f in features:
                    if isinstance(f, dict) and "name" in f:
                        names.append(f["name"])
                out = []
                for row in payload["rows"]:
                    if isinstance(row, list):
                        out.append(dict(zip(names, row)))
                return out
        # Single skill: maybe the dict IS a skill
        if "name" in payload and ("description" in payload or "procedure" in payload):
            return [payload]
        return []

    def save_skill(self, data: Dict, source: str = "user") -> Dict:
        """
        Save a new skill. Returns the registered skill info.
        Validates the skill schema before saving.
        """
        from core.skills import get_skills
        if "name" not in data:
            raise ValueError("Skill must have a 'name' field")
        skills = get_skills()
        # Use the skills manager to add (validates and saves)
        skill = skills.add(data)
        # Track as imported
        if source.startswith("imported"):
            self.imported.setdefault("imported", [])
            if skill.name not in self.imported["imported"]:
                self.imported["imported"].append({
                    "name": skill.name,
                    "source": source,
                    "added_at": time.time(),
                })
            self._save_imported()
        return {
            "name": skill.name,
            "description": skill.description,
            "triggers": skill.triggers,
            "source": source,
        }

    def learn_skill(self, name: str, description: str,
                    procedure: str, examples: Optional[List[Dict]] = None,
                    triggers: Optional[List[str]] = None) -> Dict:
        """
        'Learn' a new skill: capture a procedure the user describes.
        Simpler than save_skill — just name + description + procedure.
        """
        data = {
            "name": name,
            "description": description,
            "trigger_keywords": triggers or [],
            "input_schema": {"text": "user input"},
            "procedure": procedure,
            "examples": examples or [],
            "version": "1.0.0",
            "enabled": True,
        }
        return self.save_skill(data, source="learned:user")

    def export_skill(self, name: str) -> Dict:
        """Export a skill as a JSON-serializable dict."""
        from core.skills import get_skills
        skill = get_skills().get(name)
        if not skill:
            raise KeyError(f"Skill not found: {name}")
        return skill.data

    def learn_from_interaction(self, tab_id: str, skill_name: str,
                               description: str) -> Dict:
        """
        Capture a successful interaction as a skill.
        Pulls the recent messages from the tab as the procedure.
        """
        from core.session_state import get_session
        session = get_session()
        tab = session.get_tab(tab_id)
        if not tab:
            raise ValueError(f"Tab not found: {tab_id}")
        messages = tab.get("messages", [])
        # Build procedure from the last user+assistant exchange
        procedure_lines = ["# Auto-captured from chat interaction", ""]
        for m in messages[-6:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            procedure_lines.append(f"## {role.title()}")
            procedure_lines.append(content)
            procedure_lines.append("")
        procedure = "\n".join(procedure_lines)
        # Extract triggers from description
        triggers = re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", description.lower())
        triggers = list(dict.fromkeys(triggers))[:8]
        return self.learn_skill(
            name=skill_name,
            description=description,
            procedure=procedure,
            triggers=triggers,
        )

    def stats(self) -> Dict:
        from core.skills import get_skills
        return {
            "total_skills": len(get_skills()),
            "imported": len(self.imported.get("imported", [])),
            "dir": str(self.dir),
        }


_lib: Optional[SkillLibrary] = None


def get_library() -> SkillLibrary:
    global _lib
    if _lib is None:
        _lib = SkillLibrary()
    return _lib


if __name__ == "__main__":
    lib = get_library()
    print(lib.stats())
