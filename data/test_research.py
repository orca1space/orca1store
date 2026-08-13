import json
import sys
sys.path.insert(0, r"D:\Hermes")
from core.research_engine import dispatch

cases = [
    ("research.index_local", {"documents":[{"id":"a","text":"Offline OCR and local software engineering"},{"id":"b","text":"Local code intelligence for software teams"}]}),
    ("research.analyze_corpus", {"documents":[{"id":"a","text":"Offline OCR and local software engineering"},{"id":"b","text":"Local code intelligence for software teams"}],"limit":10}),
    ("research.market_trend", {"snapshots":[{"period":"2026-01","value":100},{"period":"2026-02","value":125},{"period":"2026-03","value":120}]}),
    ("research.intelligence_report", {"documents":[{"id":"a","text":"Offline OCR and local software engineering"},{"id":"b","text":"Local code intelligence for software teams"}],"snapshots":[{"period":"2026-01","value":100},{"period":"2026-03","value":120}]}),
]
results = {op: dispatch(op, params) for op, params in cases}
print(json.dumps(results, ensure_ascii=False, indent=2))
assert all(v.get("local_only") is True for v in results.values())
assert all(v.get("ok") is True for v in results.values())
