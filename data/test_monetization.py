import json
from pathlib import Path
import sys
sys.path.insert(0, r"D:\Hermes")
from core.monetization_engine import dispatch

cases = [
    ("monetization.score_opportunity", {"topic":"local software engineering tutorials", "audience":"developers", "competition":0.45, "commercial_intent":0.8, "evergreen":0.9}),
    ("monetization.content_brief", {"topic":"offline OCR for developers", "audience":"developers", "intent":"commercial investigation"}),
    ("monetization.policy_check", {"title":"Offline OCR guide", "body":"Original practical guide with sources and clear disclosures.", "links":["https://example.com/docs"], "disclosures":["This is an educational article."]}),
    ("monetization.metrics", {"sessions":1200, "pageviews":3000, "ad_clicks":72, "revenue":84.5}),
    ("monetization.execution_plan", {"topic":"Build and validate an offline-first tutorial site", "budget":1000, "constraints":["local-only","human approval for publish"]}),
]
results = {op: dispatch(op, params) for op, params in cases}
print(json.dumps(results, ensure_ascii=False, indent=2))
assert all(isinstance(v, dict) for v in results.values())
