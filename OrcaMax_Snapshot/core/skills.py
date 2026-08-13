"""
Hermes Skills System
A skill is a teachable procedure. Each skill is a JSON file describing:
- what it does
- when to use it
- what input it needs
- how to perform it (a prompt template or a Python callable)

Skills are stored as JSON in the skills/ directory. The registry is a manifest
of all available skills. The user can add/remove/edit skills freely.
"""
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import SKILLS_DIR, SKILLS_REGISTRY


SKILL_SCHEMA = {
    "name": str,            # unique identifier, e.g. "summarize_text"
    "description": str,     # human-readable explanation
    "trigger_keywords": list,  # words/phrases that suggest this skill is relevant
    "input_schema": dict,   # expected input fields and types
    "procedure": str,       # step-by-step instructions OR a prompt template
    "examples": list,       # optional list of {input, output} examples
    "version": str,         # semantic version
    "enabled": bool,        # can be temporarily disabled
}


class Skill:
    """Represents a single teachable skill."""

    def __init__(self, data: Dict):
        self.data = data
        self._validate()

    def _validate(self):
        for key, typ in SKILL_SCHEMA.items():
            if key not in self.data:
                if key in ("trigger_keywords", "input_schema", "examples"):
                    self.data[key] = [] if key != "input_schema" else {}
                elif key in ("enabled",):
                    self.data[key] = True
                elif key == "version":
                    self.data[key] = "1.0.0"
                else:
                    raise ValueError(f"Skill missing required field: {key}")
            elif not isinstance(self.data[key], typ):
                # Coerce simple cases
                if typ is bool and isinstance(self.data[key], str):
                    self.data[key] = self.data[key].lower() in ("true", "1", "yes")
                elif typ is list and isinstance(self.data[key], str):
                    self.data[key] = [self.data[key]]
                else:
                    raise TypeError(
                        f"Skill '{self.data.get('name', '?')}' field '{key}' "
                        f"must be {typ.__name__}, got {type(self.data[key]).__name__}"
                    )

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def description(self) -> str:
        return self.data["description"]

    @property
    def procedure(self) -> str:
        return self.data["procedure"]

    @property
    def triggers(self) -> List[str]:
        return [k.lower() for k in self.data.get("trigger_keywords", [])]

    @property
    def enabled(self) -> bool:
        return self.data.get("enabled", True)

    def matches(self, user_message: str) -> float:
        """
        Score how well this skill matches a user message (0.0 to 1.0).
        Uses keyword overlap with bonuses for exact phrase matches.
        """
        msg = user_message.lower()
        score = 0.0
        for kw in self.triggers:
            kw_l = kw.lower()
            if kw_l in msg:
                # Whole-word match is stronger
                if re.search(rf'\b{re.escape(kw_l)}\b', msg):
                    score += 0.4
                else:
                    score += 0.2
        # Description similarity (simple)
        desc_words = set(re.findall(r'\w+', self.description.lower()))
        msg_words = set(re.findall(r'\w+', msg))
        if desc_words and msg_words:
            overlap = len(desc_words & msg_words) / len(desc_words | msg_words)
            score += overlap * 0.3
        return min(score, 1.0)

    def to_prompt(self) -> str:
        """Render the skill as a prompt section for the LLM."""
        lines = [f"### Skill: {self.name}", f"Description: {self.description}"]
        if self.data.get("input_schema"):
            lines.append("Inputs:")
            for k, v in self.data["input_schema"].items():
                lines.append(f"  - {k}: {v}")
        lines.append(f"Procedure:\n{self.procedure}")
        if self.data.get("examples"):
            lines.append("Examples:")
            for ex in self.data["examples"][:3]:
                lines.append(f"  Input: {ex.get('input', '')}")
                lines.append(f"  Output: {ex.get('output', '')}")
        return "\n".join(lines)


class SkillsManager:
    """Loads, saves, and indexes all skills."""

    def __init__(self, skills_dir: Path = SKILLS_DIR, registry: Path = SKILLS_REGISTRY):
        self.skills_dir = skills_dir
        self.registry_path = registry
        self.skills: Dict[str, Skill] = {}
        self.load()

    def load(self):
        """Load all skill JSON files from the skills directory."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        for f in self.skills_dir.glob("*.json"):
            if f.name == "registry.json":
                continue
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if "name" not in data:
                    data["name"] = f.stem
                skill = Skill(data)
                self.skills[skill.name] = skill
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"[skills] Warning: skipping {f.name}: {e}")
        self._save_registry()

    def _save_registry(self):
        """Save the registry index (does not duplicate skill data)."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        reg = {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "triggers": s.triggers,
                    "enabled": s.enabled,
                    "file": f"{s.name}.json",
                }
                for s in self.skills.values()
            ],
            "updated_at": time.time(),
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)

    def add(self, skill_data: Dict) -> Skill:
        """Add or update a skill. Persists to its own JSON file."""
        if "name" not in skill_data:
            raise ValueError("Skill must have a 'name' field")
        skill = Skill(skill_data)
        path = self.skills_dir / f"{skill.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill.data, f, ensure_ascii=False, indent=2)
        self.skills[skill.name] = skill
        self._save_registry()
        return skill

    def remove(self, name: str) -> bool:
        if name not in self.skills:
            return False
        path = self.skills_dir / f"{name}.json"
        if path.exists():
            path.unlink()
        del self.skills[name]
        self._save_registry()
        return True

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def find_matching(self, user_message: str, top_k: int = 3,
                      min_score: float = 0.15) -> List[tuple]:
        """Find skills that match a user message, sorted by relevance."""
        scored = []
        for skill in self.skills.values():
            if not skill.enabled:
                continue
            score = skill.matches(user_message)
            if score >= min_score:
                scored.append((skill, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def list_all(self) -> List[Dict]:
        return [
            {"name": s.name, "description": s.description, "enabled": s.enabled,
             "triggers": s.triggers}
            for s in self.skills.values()
        ]

    def __len__(self):
        return len(self.skills)


_manager: Optional[SkillsManager] = None


def get_skills() -> SkillsManager:
    global _manager
    if _manager is None:
        _manager = SkillsManager()
    return _manager


if __name__ == "__main__":
    sm = get_skills()
    print(f"Loaded {len(sm)} skills")
    for s in sm.list_all():
        print(f"  - {s['name']}: {s['description']}")
