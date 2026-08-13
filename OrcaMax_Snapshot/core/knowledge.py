"""
Hermes Knowledge Base
Manages document ingestion and retrieval. Documents can come from:
- Plain text files
- Markdown files
- User-pasted content via the training interface

Documents are chunked (simple paragraph/sentence splitter) and stored
in the vector store with metadata.
"""
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import KNOWLEDGE_DIR, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from core.vector_store import get_store


SUPPORTED_EXT = {".txt", ".md", ".markdown", ".rst"}


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Simple chunker: splits on paragraph boundaries first, then sentences if a
    paragraph is too long. Keeps chunks semantically coherent.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?\n])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent
        else:
            if len(current) + len(para) + 2 <= max_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para

    if current:
        chunks.append(current)

    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-overlap:] if len(overlapped[-1]) > overlap else overlapped[-1]
            overlapped.append(tail + " " + chunks[i])
        chunks = overlapped

    return chunks


class KnowledgeBase:
    """High-level interface for document management and retrieval."""

    def __init__(self):
        self.store = get_store()
        self.dir = KNOWLEDGE_DIR

    def add_text(self, text: str, source: str = "user",
                 metadata: Optional[Dict] = None) -> List[str]:
        """Add raw text. Returns list of created entry IDs."""
        chunks = chunk_text(text)
        if not chunks:
            return []
        metadatas = []
        for i, c in enumerate(chunks):
            m = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "added_at": time.time(),
            }
            if metadata:
                m.update(metadata)
            metadatas.append(m)
        return self.store.add_batch(chunks, metadatas=metadatas)

    def add_file(self, path: str) -> List[str]:
        """Ingest a single file. Returns IDs of added chunks."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if p.suffix.lower() not in SUPPORTED_EXT:
            raise ValueError(f"Unsupported file type: {p.suffix}. Use one of {SUPPORTED_EXT}")
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.add_text(text, source=str(p), metadata={"filename": p.name})

    def add_directory(self, path: str, recursive: bool = True) -> Dict:
        """Ingest all supported files in a directory."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        files = list(p.rglob("*")) if recursive else list(p.glob("*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
        results = {"files_processed": 0, "chunks_added": 0, "errors": []}
        for f in files:
            try:
                ids = self.add_file(str(f))
                results["files_processed"] += 1
                results["chunks_added"] += len(ids)
            except Exception as e:
                results["errors"].append({"file": str(f), "error": str(e)})
        self.store.save()
        return results

    def search(self, query: str, top_k: int = TOP_K_RESULTS,
               threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """Returns list of {content, score, metadata}."""
        results = self.store.search(query, top_k=top_k, threshold=threshold)
        return [
            {
                "content": entry["content"],
                "score": score,
                "metadata": entry["metadata"],
                "id": entry["id"],
            }
            for entry, score in results
        ]

    def stats(self) -> Dict:
        return self.store.stats()

    def clear(self):
        self.store.clear()
        self.store.save()

    def save(self):
        self.store.save()


_kb: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


if __name__ == "__main__":
    kb = get_kb()
    print("Knowledge base:", kb.stats())
