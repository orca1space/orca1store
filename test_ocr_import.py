import json
import urllib.request

payload = json.dumps({"op": "imports.ocr", "input": "D:\\Hermes\\imported_sources\\markitdown\\packages\\markitdown-ocr\\tests\\ocr_test_data\\pdf_image_start.pdf", "output": "D:\\Hermes\\data\\ocr_test_image_output.pdf", "redo": True}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:7777/api/agent/exec", data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=600) as r:
    print(r.read().decode("utf-8", "replace")[:5000])
