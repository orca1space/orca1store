"""
Hermes Hybrid Search
Vector + keyword (BM25-style) hybrid retrieval with reranking.
Inspired by LangGraph/LangChain hybrid search.
Pure local. No external services.

Combines:
- Vector similarity (semantic) - from core/knowledge.py
- BM25 (keyword) - implemented from scratch
- Reciprocal Rank Fusion (RRF) for combining
- Cross-encoder rerank (optional, simple)
"""
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class BM25:
    """
    BM25 (Best Matching 25) - classic keyword search algorithm.
    Pure Python. No external services.
    """

    def __init__(self, documents: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.N = len(documents)
        self.doc_lens = [len(self._tokenize(d.get("content", ""))) for d in documents]
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N > 0 else 1
        # Document frequency per term
        self.df: Dict[str, int] = {}
        # Inverted index: term -> list of doc indices
        self.inverted_index: Dict[str, List[int]] = {}
        for i, doc in enumerate(documents):
            tokens = set(self._tokenize(doc.get("content", "")))
            for token in tokens:
                self.df[token] = self.df.get(token, 0) + 1
                self.inverted_index.setdefault(token, []).append(i)
        # Term frequencies per doc
        self.tf: List[Counter] = []
        for doc in documents:
            tokens = self._tokenize(doc.get("content", ""))
            self.tf.append(Counter(tokens))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-word, keep CJK chars."""
        text = text.lower()
        # Split on whitespace and punctuation, but keep CJK
        tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]|[\u0600-\u06ff]+", text)
        return [t for t in tokens if len(t) > 1]

    def score(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Score documents against query. Returns (doc_idx, score) sorted."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores: Dict[int, float] = {}
        for qt in query_tokens:
            if qt not in self.inverted_index:
                continue
            df = self.df[qt]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx in self.inverted_index[qt]:
                tf = self.tf[doc_idx].get(qt, 0)
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0) + idf * numerator / denominator
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    Combine multiple rankings using RRF.
    rankings: list of lists of doc indices, one per ranking source
    Returns dict of doc_idx -> fused score.
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0) + 1.0 / (k + rank)
    return fused


class HybridSearch:
    """Combines vector + BM25 + optional rerank for high-quality retrieval."""

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base  # core.knowledge.KnowledgeBase
        self._bm25_cache: Optional[BM25] = None
        self._bm25_query: Optional[str] = None

    def search(self, query: str, top_k: int = 10,
               vector_weight: float = 0.6, bm25_weight: float = 0.4,
               use_rrf: bool = True) -> List[Dict]:
        """
        Hybrid search combining vector + BM25.
        Returns list of {id, content, score, source, score_vector, score_bm25}.
        """
        if self.kb is None:
            return []
        # Vector search
        vector_results = self.kb.search(query, top_k=top_k * 2)
        # BM25 search
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        if use_rrf:
            # RRF combination
            vector_ranks = [r["id"] for r in vector_results]
            bm25_ranks = [r["id"] for r in bm25_results]
            fused = reciprocal_rank_fusion([vector_ranks, bm25_ranks])
            # Build result list
            all_ids = set(vector_ranks + bm25_ranks)
            id_to_doc = {}
            for r in vector_results:
                id_to_doc[r["id"]] = r
            for r in bm25_results:
                if r["id"] not in id_to_doc:
                    id_to_doc[r["id"]] = r
            results = []
            for doc_id, fused_score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                doc = id_to_doc.get(doc_id, {})
                if not doc:
                    continue
                results.append({
                    "id": doc_id,
                    "content": doc.get("content", ""),
                    "score": fused_score,
                    "score_vector": next((r["score"] for r in vector_results if r["id"] == doc_id), 0),
                    "score_bm25": next((r["score"] for r in bm25_results if r["id"] == doc_id), 0),
                    "metadata": doc.get("metadata", {}),
                })
            return results
        else:
            # Weighted score combination (need normalization)
            results = {}
            for r in vector_results:
                results[r["id"]] = {
                    "id": r["id"],
                    "content": r["content"],
                    "score": r["score"] * vector_weight,
                    "score_vector": r["score"],
                    "score_bm25": 0,
                    "metadata": r.get("metadata", {}),
                }
            for r in bm25_results:
                if r["id"] in results:
                    results[r["id"]]["score"] += r["score"] * bm25_weight
                    results[r["id"]]["score_bm25"] = r["score"]
                else:
                    results[r["id"]] = {
                        "id": r["id"],
                        "content": r["content"],
                        "score": r["score"] * bm25_weight,
                        "score_vector": 0,
                        "score_bm25": r["score"],
                        "metadata": r.get("metadata", {}),
                    }
            sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)
            return sorted_results[:top_k]

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Search using BM25 over all KB documents."""
        if not hasattr(self.kb, 'store'):
            return []
        # Get all documents
        try:
            all_entries = self.kb.store.all_entries() if hasattr(self.kb.store, 'all_entries') else []
        except Exception:
            all_entries = []
        if not all_entries:
            return []
        # Build/reuse BM25 index
        if self._bm25_cache is None or self._bm25_query != query:
            self._bm25_cache = BM25(all_entries)
            self._bm25_query = query
        ranked = self._bm25_cache.score(query, top_k=top_k)
        results = []
        for doc_idx, score in ranked:
            doc = all_entries[doc_idx]
            results.append({
                "id": doc.get("id"),
                "content": doc.get("content", ""),
                "score": score,
                "metadata": doc.get("metadata", {}),
            })
        return results

    def stats(self) -> Dict:
        return {
            "kb_loaded": self.kb is not None,
            "vector_weight": 0.6,
            "bm25_weight": 0.4,
            "method": "RRF",
        }


# Singleton
_search: Optional[HybridSearch] = None


def get_hybrid_search() -> HybridSearch:
    global _search
    if _search is None:
        from core.knowledge import get_kb
        _search = HybridSearch(knowledge_base=get_kb())
    return _search


if __name__ == "__main__":
    hs = get_hybrid_search()
    print("Hybrid search:", hs.stats())
    results = hs.search("PDF files", top_k=3)
    for r in results:
        print(f"  score={r['score']:.3f} (vec={r['score_vector']:.3f}, bm25={r['score_bm25']:.3f}) {r['content'][:50]}")
