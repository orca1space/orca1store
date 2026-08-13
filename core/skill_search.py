"""
Hermes Skill Search
Semantic search over skills using the same embedding model as the KB.
Pure local. No external services.

Each skill is embedded as: name + description + triggers.
On a query, we embed the query and find the top-K closest skills by cosine similarity.
This works much better than keyword matching for selecting the right skill
from a large library (94+ skills).

Cache: D:\\Hermes\\data\\skill_embeddings.json
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT


EMBED_CACHE_PATH = HERMES_ROOT / "data" / "skill_embeddings.json"


class SkillSearch:
    """Semantic skill search using embeddings."""

    def __init__(self, cache_path: Path = EMBED_CACHE_PATH):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = self._load_cache()
        self._embedder = None
        self._last_indexed_at: Dict[str, float] = {}

    def _load_cache(self) -> Dict:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except OSError:
            pass

    def _get_embedder(self):
        if self._embedder is None:
            from core.llm import get_llm
            self._embedder = get_llm()._get_embedder()
        return self._embedder

    def _skill_to_text(self, skill) -> str:
        """Build a search-friendly text representation of a skill."""
        parts = [skill.name.replace("_", " ")]
        if skill.description:
            parts.append(skill.description)
        for kw in skill.triggers:
            if kw and len(kw) > 2:
                parts.append(kw.replace("_", " "))
        # Deduplicate
        seen = set()
        result = []
        for p in parts:
            pl = p.lower().strip()
            if pl and pl not in seen:
                seen.add(pl)
                result.append(p)
        return " | ".join(result)

    def _cosine(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def index_skills(self, skills: Dict[str, "Skill"], force: bool = False) -> int:
        """Embed all skills and store in cache. Returns count indexed."""
        embedder = self._get_embedder()
        count = 0
        for name, skill in skills.items():
            if not force and name in self._cache:
                continue
            text = self._skill_to_text(skill)
            try:
                vec = embedder.encode([text], show_progress_bar=False,
                                      convert_to_numpy=True)[0].tolist()
                self._cache[name] = vec
                count += 1
            except Exception as e:
                print(f"[skill_search] Failed to embed {name}: {e}")
        self._save_cache()
        return count

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.25) -> List[Tuple[str, float, "Skill"]]:
        """
        Find skills most relevant to the query.
        Returns list of (skill_name, score, skill_object).
        """
        from core.skills import get_skills
        skills = get_skills().skills
        if not skills:
            return []
        # Re-index if cache is empty or stale (new skills added)
        if not self._cache or any(name not in self._cache for name in skills.keys()):
            self.index_skills(skills)

        # Embed query
        try:
            embedder = self._get_embedder()
            qvec = embedder.encode([query], show_progress_bar=False,
                                   convert_to_numpy=True)[0].tolist()
        except Exception as e:
            print(f"[skill_search] Query embed failed: {e}")
            return []

        scored = []
        for name, vec in self._cache.items():
            if name not in skills:
                continue
            sim = self._cosine(qvec, vec)
            if sim >= min_score:
                scored.append((name, sim, skills[name]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def stats(self) -> Dict:
        return {
            "cached_embeddings": len(self._cache),
            "cache_path": str(self.cache_path),
        }


_search: Optional[SkillSearch] = None


def get_skill_search() -> SkillSearch:
    global _search
    if _search is None:
        _search = SkillSearch()
    return _search


if __name__ == "__main__":
    s = get_skill_search()
    print("Skill search:", s.stats())
