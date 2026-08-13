import json
import sys
sys.path.insert(0, r"D:\Hermes")
from core.workflow_engine import dispatch

result = dispatch("workflow.adsense_simulation", {
    "topic":"offline OCR for developers",
    "audience":"developers",
    "keywords":["offline OCR", "developer OCR"],
    "competition":0.45,
    "commercial_intent":0.8,
    "evergreen":0.9,
    "documents":[{"id":"local-notes","text":"Offline OCR improves document workflows for software teams."}],
    "snapshots":[{"period":"2026-01","value":100},{"period":"2026-02","value":125},{"period":"2026-03","value":120}],
    "metrics":{"impressions":1000,"clicks":30,"revenue":42.5}
})
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result.get("ok") is True
assert result.get("local_only") is True
assert result.get("publication_performed") is False
assert result.get("next_required_action") == "human_approval_before_publish"
assert result["workflow"]["approval"]["approval"]["status"] == "pending"
