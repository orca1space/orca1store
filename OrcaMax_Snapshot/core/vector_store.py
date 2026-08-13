"""
Hermes Vector Store
Lightweight in-process vector store using numpy. No external dependencies
beyond numpy. Persisted to JSON on disk.

Designed for personal-scale knowledge bases (hundreds to a few thousand docs).
"""
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import VECTOR_STORE_PATH
from core.llm import get_llm


class VectorStore:
    """
    Simple vector store with cosine similarity search.
    Each entry: {id, content, embedding, metadata, created_at}
    """

    def __init__(self, path: Path = VECTOR_STORE_PATH):
        self.path = path
        self.entries: List[Dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = False
        self.load()

    def load(self):
        """Load from disk. Initializes empty if file doesn't exist."""
        if not self.path.exists():
            self.entries = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.entries = raw.get("entries", [])
            # Reconstruct numpy matrix
            if self.entries:
                embeddings = [e["embedding"] for e in self.entries]
                # Handle variable dim (older entries may differ)
                dim = len(embeddings[0]) if embeddings else 0
                self._matrix = np.array(embeddings, dtype=np.float32)
            else:
                self._matrix = None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[vector_store] Warning: could not load {self.path}: {e}. Starting fresh.")
            self.entries = []

    def save(self):
        """Persist to disk. Only writes if there are unsaved changes."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": self.entries, "saved_at": time.time()}, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
        self._dirty = False

    def _rebuild_matrix(self):
        if not self.entries:
            self._matrix = None
            return
        embeddings = [e["embedding"] for e in self.entries]
        self._matrix = np.array(embeddings, dtype=np.float32)

    def add(self, content: str, metadata: Optional[Dict] = None,
            embedding: Optional[List[float]] = None) -> str:
        """
        Add a document. If embedding not provided, computes it via LLM.
        Returns the assigned id.
        """
        if embedding is None:
            embedding = get_llm().embed(content)
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        self.entries.append(entry)
        self._rebuild_matrix()
        self._dirty = True
        return entry_id

    def add_batch(self, contents: List[str], metadatas: Optional[List[Dict]] = None,
                  embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """Add many docs at once. Computes embeddings in batch if not provided."""
        if embeddings is None:
            embeddings = get_llm().embed_batch(contents)
        if metadatas is None:
            metadatas = [{} for _ in contents]
        ids = []
        now = time.time()
        new_entries = []
        for content, emb, meta in zip(contents, embeddings, metadatas):
            entry_id = str(uuid.uuid4())
            ids.append(entry_id)
            new_entries.append({
                "id": entry_id,
                "content": content,
                "embedding": emb,
                "metadata": meta,
                "created_at": now,
            })
        self.entries.extend(new_entries)
        self._rebuild_matrix()
        self._dirty = True
        return ids

    def search(self, query: str, top_k: int = 5,
               threshold: float = 0.0,
               filter_metadata: Optional[Dict] = None) -> List[Tuple[Dict, float]]:
        """
        Semantic search. Returns list of (entry, similarity_score) sorted by score desc.
        """
        if not self.entries or self._matrix is None:
            return []
        query_emb = np.array(get_llm().embed(query), dtype=np.float32)
        # Normalize for cosine similarity
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        m_norm = self._matrix / (np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-10)
        sims = m_norm @ q_norm  # cosine similarities
        # Filter
        if filter_metadata:
            mask = np.array([
                all(entry["metadata"].get(k) == v for k, v in filter_metadata.items())
                for entry in self.entries
            ])
            sims = np.where(mask, sims, -np.inf)
        # Top-k
        k = min(top_k, len(self.entries))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        results = []
        for i in idx:
            score = float(sims[i])
            if score >= threshold:
                results.append((self.entries[int(i)], score))
        return results

    def delete(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        if len(self.entries) < before:
            self._rebuild_matrix()
            self._dirty = True
            return True
        return False

    def clear(self):
        self.entries = []
        self._matrix = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self.entries)

    def stats(self) -> Dict:
        dim = len(self.entries[0]["embedding"]) if self.entries else 0
        total_chars = sum(len(e["content"]) for e in self.entries)
        return {
            "count": len(self.entries),
            "dim": dim,
            "total_chars": total_chars,
            "file_size_kb": round(self.path.stat().st_size / 1024, 2) if self.path.exists() else 0,
        }


# Module-level singleton
_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


if __name__ == "__main__":
    s = get_store()
    print("Store stats:", s.stats())
