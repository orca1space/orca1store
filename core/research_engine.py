from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_ROOTS = (ROOT.resolve(), (ROOT / "data").resolve(), (ROOT / "imported_sources").resolve())
_STOPWORDS = set("the and for with that this from are was were into your have has had not you our their about using local only من في على الى عن هذا هذه هو هي مع كان تكون يتم الى و من على".split())
_WORD_RE = re.compile(r"[A-Za-z0-9_\u0600-\u06ff-]{3,}")


def _safe_path(value: Any) -> Path:
    p = Path(str(value)).expanduser().resolve()
    if not any(p == root or root in p.parents for root in _ALLOWED_ROOTS):
        raise ValueError("path_outside_local_roots")
    return p


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS]


def _read_documents(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in params.get("documents", []) or []:
        if isinstance(item, dict):
            docs.append({"id": str(item.get("id", len(docs) + 1)), "text": str(item.get("text", ""))})
        else:
            docs.append({"id": str(len(docs) + 1), "text": str(item)})
    for raw in params.get("paths", []) or []:
        p = _safe_path(raw)
        if p.is_file() and p.stat().st_size <= 5_000_000:
            docs.append({"id": str(p), "text": p.read_text(encoding="utf-8", errors="ignore")})
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in {".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".ts"} and child.stat().st_size <= 5_000_000:
                    docs.append({"id": str(child), "text": child.read_text(encoding="utf-8", errors="ignore")})
    return docs


def index_local(params: Dict[str, Any]) -> Dict[str, Any]:
    docs = _read_documents(params)
    rows = []
    for doc in docs:
        words = _words(doc["text"])
        digest = hashlib.sha256(doc["text"].encode("utf-8", "ignore")).hexdigest()
        rows.append({"id": doc["id"], "characters": len(doc["text"]), "tokens": len(words), "sha256": digest, "top_terms": Counter(words).most_common(10)})
    return {"ok": True, "local_only": True, "document_count": len(rows), "documents": rows}


def analyze_corpus(params: Dict[str, Any]) -> Dict[str, Any]:
    docs = _read_documents(params)
    counts = Counter()
    for doc in docs:
        counts.update(_words(doc["text"]))
    total = sum(counts.values())
    terms = [{"term": term, "count": count, "share_percent": round(count * 100 / total, 4) if total else 0.0} for term, count in counts.most_common(int(params.get("limit", 20) or 20))]
    return {"ok": True, "local_only": True, "document_count": len(docs), "total_tokens": total, "top_terms": terms, "method": "frequency_with_stopword_filter"}


def market_trend(params: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = params.get("snapshots", []) or []
    if not snapshots:
        return {"ok": False, "local_only": True, "error": "snapshots_required"}
    series = []
    for snap in snapshots:
        if not isinstance(snap, dict):
            continue
        value = float(snap.get("value", 0) or 0)
        series.append({"period": str(snap.get("period", len(series) + 1)), "value": value})
    if not series:
        return {"ok": False, "local_only": True, "error": "valid_snapshots_required"}
    first, last = series[0]["value"], series[-1]["value"]
    delta = last - first
    pct = round((delta / abs(first)) * 100, 4) if first else None
    direction = "rising" if delta > 0 else "falling" if delta < 0 else "flat"
    mean = sum(x["value"] for x in series) / len(series)
    variance = sum((x["value"] - mean) ** 2 for x in series) / len(series)
    return {"ok": True, "local_only": True, "trend": {"direction": direction, "change": round(delta, 4), "change_percent": pct, "average": round(mean, 4), "volatility": round(math.sqrt(variance), 4), "series": series}, "interpretation": "descriptive_local_snapshot_analysis"}


def intelligence_report(params: Dict[str, Any]) -> Dict[str, Any]:
    corpus = analyze_corpus(params)
    trend = market_trend({"snapshots": params.get("snapshots", [])}) if params.get("snapshots") else {"ok": True, "local_only": True, "trend": None}
    return {"ok": corpus.get("ok", False) and trend.get("ok", False), "local_only": True, "report": {"corpus": corpus, "trend": trend, "limitations": ["no_external_market_data", "descriptive_not_predictive", "requires_human_review_before_business_decision"]}}


_HANDLERS = {"research.index_local": index_local, "research.analyze_corpus": analyze_corpus, "research.market_trend": market_trend, "research.intelligence_report": intelligence_report}


def dispatch(operation: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        handler = _HANDLERS.get(operation)
        if handler is None:
            return {"ok": False, "local_only": True, "error": "unknown_operation", "operation": operation}
        return handler(params or {})
    except (OSError, ValueError, TypeError) as exc:
        return {"ok": False, "local_only": True, "error": type(exc).__name__, "detail": str(exc)}
