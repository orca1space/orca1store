import json
import sys
sys.path.insert(0, r"D:\Hermes")
from core.content_engine import dispatch

brief = dispatch("content.generate_draft", {"topic":"offline OCR for developers", "audience":"developers", "search_intent":"commercial investigation", "keywords":["offline OCR", "developer OCR"]})
optimized = dispatch("content.seo_optimize", {"draft": brief.get("draft", {}), "keywords":["offline OCR", "developer OCR"]})
validated = dispatch("content.validate_draft", {"draft": brief.get("draft", {})})
result = {"generate": brief, "optimize": optimized, "validate": validated}
print(json.dumps(result, ensure_ascii=False, indent=2))
assert brief.get("ok") and optimized.get("ok") and validated.get("ok")
assert validated.get("requires_human_review") is True
