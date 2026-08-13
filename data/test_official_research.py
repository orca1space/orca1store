import json
import sys
from pathlib import Path
sys.path.insert(0, r"D:\Hermes")
from core.research_engine import dispatch

path = Path(r"D:\Hermes\data\google_publisher_research.md")
doc = {"id": path.name, "text": path.read_text(encoding="utf-8")}
result = dispatch("research.intelligence_report", {"documents":[doc]})
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result.get("ok") is True
assert result.get("local_only") is True
assert result["report"]["corpus"]["document_count"] == 1
assert result["report"]["corpus"]["total_tokens"] > 100
