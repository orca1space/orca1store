"""
Hermes Memory
Persists conversation history and training sessions.
Used to:
- Provide context in ongoing conversations
- Store lessons learned during training
- Allow the user to review and revise past interactions
"""
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import MEMORY_PATH


class Memory:
    """Persists conversations and training sessions."""

    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        self.data = {
            "conversations": [],     # list of session objects
            "training_sessions": [], # list of training objects
            "lessons": [],           # distilled lessons from training
        }
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key in self.data:
                if key in raw:
                    self.data[key] = raw[key]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[memory] Warning: could not load memory: {e}. Starting fresh.")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    # Conversations
    def new_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation. Optionally specify the ID (used by tabs)."""
        conv_id = conversation_id or str(uuid.uuid4())
        # Avoid duplicates
        for conv in self.data["conversations"]:
            if conv["id"] == conv_id:
                return conv_id
        self.data["conversations"].append({
            "id": conv_id,
            "started_at": time.time(),
            "messages": [],
            "training_mode": False,
        })
        self.save()
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str,
                    metadata: Optional[Dict] = None) -> None:
        for conv in self.data["conversations"]:
            if conv["id"] == conv_id:
                conv["messages"].append({
                    "role": role,
                    "content": content,
                    "timestamp": time.time(),
                    "metadata": metadata or {},
                })
                self.save()
                return
        # Auto-create the conversation if it doesn't exist (tabs system)
        self.new_conversation(conversation_id=conv_id)
        # Retry
        for conv in self.data["conversations"]:
            if conv["id"] == conv_id:
                conv["messages"].append({
                    "role": role,
                    "content": content,
                    "timestamp": time.time(),
                    "metadata": metadata or {},
                })
                self.save()
                return

    def get_conversation(self, conv_id: str) -> Optional[Dict]:
        for conv in self.data["conversations"]:
            if conv["id"] == conv_id:
                return conv
        return None

    def get_recent_messages(self, conv_id: str, n: int = 10) -> List[Dict]:
        conv = self.get_conversation(conv_id)
        if not conv:
            return []
        return conv["messages"][-n:]

    # Training sessions
    def record_training(self, topic: str, instructions: str,
                        examples: List[Dict] = None,
                        notes: str = "") -> str:
        """Record a training session (a teaching episode)."""
        session_id = str(uuid.uuid4())
        self.data["training_sessions"].append({
            "id": session_id,
            "topic": topic,
            "instructions": instructions,
            "examples": examples or [],
            "notes": notes,
            "timestamp": time.time(),
        })
        self.save()
        return session_id

    # Lessons (distilled insights from training or conversations)
    def add_lesson(self, lesson: str, source: str = "training",
                   metadata: Optional[Dict] = None) -> str:
        lesson_id = str(uuid.uuid4())
        self.data["lessons"].append({
            "id": lesson_id,
            "lesson": lesson,
            "source": source,
            "metadata": metadata or {},
            "created_at": time.time(),
        })
        self.save()
        return lesson_id

    def get_lessons(self, limit: int = 50) -> List[Dict]:
        return self.data["lessons"][-limit:]

    def stats(self) -> Dict:
        return {
            "conversations": len(self.data["conversations"]),
            "total_messages": sum(len(c["messages"]) for c in self.data["conversations"]),
            "training_sessions": len(self.data["training_sessions"]),
            "lessons": len(self.data["lessons"]),
        }


_memory: Optional[Memory] = None


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory


if __name__ == "__main__":
    m = get_memory()
    print("Memory stats:", m.stats())
