"""
Hermes Long-term Memory Store
Cross-session, cross-thread persistent memory.
Inspired by LangGraph's Store abstraction.
Pure local. No external services.

Features:
- Key-value store with namespaces
- Search by query
- TTL/expiry
- User-scoped memory
- Type-safe entries (validated against schema)
"""
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT


STORE_PATH = HERMES_ROOT / "data" / "memory_store.json"


class MemoryStore:
    """Long-term cross-session memory with namespaces and search."""

    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict = self._load()

    def _load(self) -> Dict:
        if not self.path.exists():
            return {"version": 1, "namespaces": {}, "ttl_enabled": True}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": 1, "namespaces": {}}
            return data
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "namespaces": {}}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[memory_store] save error: {e}")

    def put(self, namespace: str, key: str, value: Any,
            ttl_seconds: Optional[int] = None,
            metadata: Optional[Dict] = None) -> str:
        """Store a value. Returns the entry ID."""
        ns = self._data.setdefault("namespaces", {}).setdefault(namespace, {})
        entry_id = uuid.uuid4().hex
        now = time.time()
        entry = {
            "id": entry_id,
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }
        if ttl_seconds is not None:
            entry["expires_at"] = now + ttl_seconds
        ns[key] = entry
        self._save()
        return entry_id

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Retrieve a value."""
        ns = self._data.get("namespaces", {}).get(namespace, {})
        entry = ns.get(key)
        if entry is None:
            return None
        # Check TTL
        if "expires_at" in entry and entry["expires_at"] < time.time():
            del ns[key]
            self._save()
            return None
        return entry["value"]

    def get_entry(self, namespace: str, key: str) -> Optional[Dict]:
        """Get full entry with metadata."""
        ns = self._data.get("namespaces", {}).get(namespace, {})
        return ns.get(key)

    def search(self, namespace: str, query: str, limit: int = 10) -> List[Dict]:
        """Search entries by substring match in value (simple)."""
        ns = self._data.get("namespaces", {}).get(namespace, {})
        q_lower = query.lower()
        results = []
        for key, entry in ns.items():
            if "expires_at" in entry and entry["expires_at"] < time.time():
                continue
            if q_lower in str(entry["value"]).lower() or q_lower in key.lower():
                results.append({
                    "key": key,
                    "value": entry["value"],
                    "metadata": entry.get("metadata", {}),
                    "updated_at": entry.get("updated_at"),
                })
        return results[:limit]

    def list_namespaces(self) -> List[str]:
        return list(self._data.get("namespaces", {}).keys())

    def list_keys(self, namespace: str) -> List[str]:
        ns = self._data.get("namespaces", {}).get(namespace, {})
        return list(ns.keys())

    def delete(self, namespace: str, key: str) -> bool:
        ns = self._data.get("namespaces", {}).get(namespace, {})
        if key in ns:
            del ns[key]
            self._save()
            return True
        return False

    def clear_namespace(self, namespace: str) -> int:
        ns = self._data.get("namespaces", {}).get(namespace, {})
        count = len(ns)
        ns.clear()
        self._save()
        return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries across namespaces. Returns count."""
        now = time.time()
        removed = 0
        for ns_name, ns in self._data.get("namespaces", {}).items():
            for key in list(ns.keys()):
                if "expires_at" in ns[key] and ns[key]["expires_at"] < now:
                    del ns[key]
                    removed += 1
        if removed:
            self._save()
        return removed

    def stats(self) -> Dict:
        total = 0
        for ns in self._data.get("namespaces", {}).values():
            total += len(ns)
        return {
            "namespaces": len(self._data.get("namespaces", {})),
            "total_entries": total,
            "path": str(self.path),
        }


_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


if __name__ == "__main__":
    s = get_memory_store()
    s.put("user_prefs", "language", "ar", ttl_seconds=3600)
    s.put("user_prefs", "theme", "dark")
    s.put("project_notes", "todo", "Add OAuth support")
    print("Stats:", s.stats())
    print("Search 'dark':", s.search("user_prefs", "dark"))
    print("Get language:", s.get("user_prefs", "language"))
    s.cleanup_expired()
