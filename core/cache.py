"""
Hermes Response Cache
Persistent local cache for LLM responses.
Pure JSON file. No external dependencies. Fully self-contained.

Used to make repeat queries instant. Hashes the user message (normalized),
stores the response, and returns it on cache hit without calling the LLM.

Cache structure (JSON):
{
  "version": 1,
  "entries": {
    "<sha256-hash>": {
      "query": "<original normalized text>",
      "response": "<LLM response>",
      "hits": 3,
      "created_at": 1234567890.0,
      "last_hit": 1234567890.0
    }
  }
}
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.logging_setup import get_logger

log = get_logger("cache")
from core.config import HERMES_ROOT


CACHE_PATH = HERMES_ROOT / "data" / "response_cache.json"
MAX_ENTRIES = 5000  # cap to avoid unbounded growth
CACHE_VERSION = 1


def _normalize(text: str) -> str:
    """Normalize text for caching. Whitespace + case insensitive."""
    return " ".join(text.lower().split())


def _key(text: str) -> str:
    """SHA-256 of normalized text."""
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


class ResponseCache:
    """Local persistent response cache."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict = self._load()

    def _load(self) -> Dict:
        if not self.path.exists():
            return {"version": CACHE_VERSION, "entries": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != CACHE_VERSION:
                return {"version": CACHE_VERSION, "entries": {}}
            return data
        except (json.JSONDecodeError, OSError):
            return {"version": CACHE_VERSION, "entries": {}}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            log.warning("Failed to save cache: %s", e)

    def get(self, query: str) -> Optional[str]:
        """Returns cached response or None."""
        key = _key(query)
        entry = self._data["entries"].get(key)
        if entry is None:
            return None
        entry["hits"] = entry.get("hits", 0) + 1
        entry["last_hit"] = time.time()
        # Persist asynchronously (best-effort)
        try:
            self._save()
        except Exception as e:
            log.debug("Cache save on hit failed (non-fatal): %s", e)
        return entry.get("response")

    def put(self, query: str, response: str):
        """Store a response. Enforces size cap (evicts oldest by last_hit)."""
        key = _key(query)
        now = time.time()
        self._data["entries"][key] = {
            "query": _normalize(query),
            "response": response,
            "hits": 0,
            "created_at": now,
            "last_hit": now,
        }
        # Enforce cap
        if len(self._data["entries"]) > MAX_ENTRIES:
            self._evict()
        self._save()

    def _evict(self):
        """Evict 10% of least-recently-hit entries."""
        entries = self._data["entries"]
        if not entries:
            return
        evict_count = max(1, len(entries) // 10)
        sorted_keys = sorted(
            entries.keys(),
            key=lambda k: entries[k].get("last_hit", 0),
        )
        for k in sorted_keys[:evict_count]:
            del entries[k]

    def stats(self) -> Dict:
        entries = self._data["entries"]
        total_hits = sum(e.get("hits", 0) for e in entries.values())
        return {
            "size": len(entries),
            "max": MAX_ENTRIES,
            "total_hits": total_hits,
            "path": str(self.path),
        }

    def clear(self):
        self._data = {"version": CACHE_VERSION, "entries": {}}
        self._save()


_cache: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache


if __name__ == "__main__":
    c = get_cache()
    print("Cache:", c.stats())
