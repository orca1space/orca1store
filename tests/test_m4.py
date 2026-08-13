"""M4: End-to-end orchestrator test with real LLM."""
import sys
import time
from pathlib import Path
HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

print("=" * 60)
print(" M4: Orchestrator End-to-End Test")
print("=" * 60)

from core.orchestrator import get_orchestrator
from core.skills import get_skills
from core.knowledge import get_kb

orch = get_orchestrator()
print("\n[1] Status BEFORE teaching:")
status = orch.status()
print(f"  KB chunks: {status['knowledge_base']['count']}")
print(f"  Skills: {status['skills']['count']}")
print(f"  Lessons: {status['memory']['lessons']}")
print(f"  LLM healthy: {status['llm']['healthy']}")

# Warm up the LLM by doing one chat (triggers lazy load)
print("\n[2] Loading LLM (first call)...")
t0 = time.time()
warmup = orch.chat("ping", stream=False)
print(f"  Warmup took {time.time()-t0:.1f}s, response: {warmup[:80]}")

print("\n[3] Teaching the agent...")
orch.teach_lesson("Always respond in Arabic when user writes in Arabic.")
orch.teach_lesson("When asked who you are, say you are Hermes, a local assistant created by the user.")

teaching_text = """
Hermes is a local AI agent that runs entirely on the user's machine.
It uses llama-cpp-python to run open-source language models.
The model is Qwen 2.5 3B Instruct in GGUF format.
Knowledge is stored as embeddings in a vector store using numpy.
The user teaches Hermes by adding skills, lessons, and documents.
Hermes is independent of any cloud service or vendor.
"""
orch.teach_document(teaching_text, source="intro")
print(f"  Taught 2 lessons + 1 document")

skill_data = {
    "name": "greet_user",
    "description": "Greets the user warmly in their language",
    "trigger_keywords": ["greet", "مرحبا", "سلام", "hello", "hi"],
    "procedure": "1. Detect user's language\n2. Compose a warm greeting in that language\n3. Be brief and friendly",
    "examples": [{"input": "مرحبا", "output": "أهلاً وسهلاً بك!"}],
    "enabled": True,
}
orch.teach_skill(skill_data)
print(f"  Taught skill: greet_user")

print("\n[4] Status AFTER teaching:")
status = orch.status()
print(f"  KB chunks: {status['knowledge_base']['count']}")
print(f"  Skills: {status['skills']['count']}")
print(f"  Lessons: {status['memory']['lessons']}")

print("\n[5] Test 1: Question in Arabic (should use Arabic + knowledge)")
t0 = time.time()
response = orch.chat("ما هو هيرمس؟", stream=False)
latency = time.time() - t0
print(f"  Q: ما هو هيرمس؟")
print(f"  A ({latency:.1f}s): {response[:400]}")
has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in response)
has_relevant = "وكيل" in response or "محلي" in response or "نموذج" in response
print(f"  Has Arabic: {has_arabic}")
print(f"  Mentions relevant concept: {has_relevant}")

print("\n[6] Test 2: Greeting (should trigger greet_user skill)")
t0 = time.time()
response = orch.chat("مرحبا كيف حالك؟", stream=False)
latency = time.time() - t0
print(f"  Q: مرحبا كيف حالك؟")
print(f"  A ({latency:.1f}s): {response[:300]}")

print("\n[7] Test 3: Question in English (should respond in English)")
t0 = time.time()
response = orch.chat("What model do you use?", stream=False)
latency = time.time() - t0
print(f"  Q: What model do you use?")
print(f"  A ({latency:.1f}s): {response[:300]}")

print("\n[8] Test 4: Knowledge base search")
results = orch.kb.search("what technology does Hermes use", top_k=3)
print(f"  Search 'what technology does Hermes use':")
for i, r in enumerate(results, 1):
    print(f"    [{r['score']:.3f}] {r['content'][:100]}...")

print("\n" + "=" * 60)
print(" M4 ✅ END-TO-END TEST COMPLETE")
print("=" * 60)
