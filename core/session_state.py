"""
Hermes Session State
Persists multi-tab conversation state across restarts.
Pure local JSON. No external services.

Stored at D:\\Hermes\\data\\session_state.json

Each tab is a separate conversation thread. The session file holds:
- tabs[]: list of {id, title, messages[], last_active}
- active_tab_id: which tab is currently shown

This is multi-session: user can open many tabs and switch between them.
"""
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT


STATE_PATH = HERMES_ROOT / "data" / "session_state.json"
MAX_TABS = 20  # cap to avoid unbounded growth
MAX_RESTORED_MESSAGES = 50  # per tab


class SessionState:
    """Manages persistent multi-tab session state."""

    def __init__(self, path: Path = STATE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict = self._load()

    def _load(self) -> Dict:
        if not self.path.exists():
            return self._default()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return self._default()
            # Migrate old single-session format
            if "tabs" not in data:
                tabs = []
                if data.get("last_conversation_id"):
                    tabs.append({
                        "id": data["last_conversation_id"],
                        "title": "Restored chat",
                        "messages": data.get("messages", []),
                        "last_active": data.get("last_active"),
                        "created_at": data.get("last_active") or time.time(),
                    })
                return {
                    "version": 2,
                    "tabs": tabs,
                    "active_tab_id": data.get("last_conversation_id"),
                }
            return {**self._default(), **data}
        except (json.JSONDecodeError, OSError):
            return self._default()

    def _default(self) -> Dict:
        return {
            "version": 2,
            "tabs": [],
            "active_tab_id": None,
        }

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[session] Warning: failed to save: {e}")

    # === TAB OPERATIONS ===

    def list_tabs(self) -> List[Dict]:
        """Return all tabs (lightweight: id, title, last_active, message_count)."""
        return [
            {
                "id": t["id"],
                "title": t.get("title", "New chat"),
                "last_active": t.get("last_active"),
                "message_count": len(t.get("messages", [])),
                "created_at": t.get("created_at"),
            }
            for t in self._data.get("tabs", [])
        ]

    def get_tab(self, tab_id: str) -> Optional[Dict]:
        """Get full tab (including messages)."""
        for t in self._data.get("tabs", []):
            if t["id"] == tab_id:
                return t
        return None

    def create_tab(self, title: Optional[str] = None) -> Dict:
        """Create a new tab. Returns the new tab summary."""
        if len(self._data.get("tabs", [])) >= MAX_TABS:
            # Remove oldest inactive tab
            tabs_sorted = sorted(
                self._data["tabs"],
                key=lambda x: x.get("last_active") or 0,
            )
            if tabs_sorted:
                self._data["tabs"].remove(tabs_sorted[0])
        tab_id = uuid.uuid4().hex
        now = time.time()
        new_tab = {
            "id": tab_id,
            "title": title or "New chat",
            "messages": [],
            "last_active": now,
            "created_at": now,
        }
        if "tabs" not in self._data:
            self._data["tabs"] = []
        self._data["tabs"].append(new_tab)
        self._data["active_tab_id"] = tab_id
        self._save()
        return {
            "id": tab_id,
            "title": new_tab["title"],
            "last_active": now,
            "message_count": 0,
            "created_at": now,
        }

    def close_tab(self, tab_id: str) -> bool:
        """Remove a tab. If it was active, switch to another."""
        tabs = self._data.get("tabs", [])
        for i, t in enumerate(tabs):
            if t["id"] == tab_id:
                tabs.pop(i)
                if self._data.get("active_tab_id") == tab_id:
                    if tabs:
                        self._data["active_tab_id"] = tabs[max(0, i - 1)]["id"]
                    else:
                        self._data["active_tab_id"] = None
                self._save()
                return True
        return False

    def switch_tab(self, tab_id: str) -> Optional[Dict]:
        """Set active tab. Returns the tab data."""
        tab = self.get_tab(tab_id)
        if tab is not None:
            self._data["active_tab_id"] = tab_id
            tab["last_active"] = time.time()
            self._save()
        return tab

    def get_active_tab(self) -> Optional[Dict]:
        """Returns the currently active tab (with messages)."""
        aid = self._data.get("active_tab_id")
        if aid is None:
            # Auto-create first tab if none exist
            if not self._data.get("tabs"):
                created = self.create_tab()
                return self.get_tab(created["id"])
            # Pick the first one
            self._data["active_tab_id"] = self._data["tabs"][0]["id"]
            self._save()
            aid = self._data["active_tab_id"]
        return self.get_tab(aid)

    def get_active_tab_id(self) -> Optional[str]:
        aid = self._data.get("active_tab_id")
        if aid is None and self._data.get("tabs"):
            self._data["active_tab_id"] = self._data["tabs"][0]["id"]
            self._save()
            aid = self._data["active_tab_id"]
        return aid

    def append_message(self, tab_id: str, role: str, content: str):
        """Append a message to a specific tab."""
        tab = self.get_tab(tab_id)
        if tab is None:
            return False
        tab["messages"].append({
            "role": role,
            "content": content,
            "ts": time.time(),
        })
        # Trim
        tab["messages"] = tab["messages"][-MAX_RESTORED_MESSAGES:]
        tab["last_active"] = time.time()
        # Auto-title from first user message
        if role == "user" and (tab.get("title") in (None, "New chat", "Restored chat")):
            text = content.strip()
            if text:
                tab["title"] = text[:40].replace("\n", " ")
                if len(text) > 40:
                    tab["title"] += "..."
        self._save()
        return True

    def set_messages(self, tab_id: str, messages: List[Dict]):
        """Replace all messages in a tab (used after full chat turn)."""
        tab = self.get_tab(tab_id)
        if tab is None:
            return False
        tab["messages"] = messages[-MAX_RESTORED_MESSAGES:]
        tab["last_active"] = time.time()
        self._save()
        return True

    def rename_tab(self, tab_id: str, title: str) -> bool:
        tab = self.get_tab(tab_id)
        if tab is None:
            return False
        tab["title"] = title[:50]
        self._save()
        return True

    # === STATE ===

    def restore(self) -> Dict:
        """Return full state for client restore."""
        return {
            "version": self._data.get("version", 2),
            "tabs": [
                {
                    "id": t["id"],
                    "title": t.get("title", "New chat"),
                    "messages": list(t.get("messages", [])),
                    "last_active": t.get("last_active"),
                    "created_at": t.get("created_at"),
                }
                for t in self._data.get("tabs", [])
            ],
            "active_tab_id": self._data.get("active_tab_id"),
        }

    def clear(self):
        """Wipe all state (start fresh)."""
        self._data = self._default()
        self._save()

    def stats(self) -> Dict:
        return {
            "version": self._data.get("version", 2),
            "tab_count": len(self._data.get("tabs", [])),
            "active_tab_id": self._data.get("active_tab_id"),
            "path": str(self.path),
        }


_state: Optional[SessionState] = None


def get_session() -> SessionState:
    global _state
    if _state is None:
        _state = SessionState()
    return _state


if __name__ == "__main__":
    s = get_session()
    print("Session state:", s.stats())
