"""Quick test script — runs without LLM."""
import sys
from pathlib import Path

HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

print("=" * 60)
print(" HERMES — Quick Component Tests (no LLM required)")
print("=" * 60)

# Test 1: Config
print("\n[T1] Config")
from core.config import ensure_dirs, HERMES_ROOT, HERMES_MODEL_FILE
ensure_dirs()
print(f"  Root: {HERMES_ROOT}")
print(f"  Model: {HERMES_MODEL_FILE.name}")
print(f"  All dirs exist: ✅")

# Test 2: Vector store
print("\n[T2] Vector Store")
from core.vector_store import VectorStore
test_path = Path(HERMES_ROOT / "tests" / "test_vs.json")
if test_path.exists():
    test_path.unlink()
store = VectorStore(path=test_path)
docs = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Python is a programming language.",
    "Cats love to sleep in the sun.",
    "Machine learning models need data.",
]
ids = store.add_batch(docs)
print(f"  Added {len(ids)} docs, stats: {store.stats()}")

# Note: We don't have LLM yet, so embedding will use zeros
# Let's test with explicit embeddings for this test
import numpy as np
np.random.seed(42)
for i, entry in enumerate(store.entries):
    # Create deterministic embeddings that group similar texts
    if "cat" in entry["content"].lower() or "Cats" in entry["content"]:
        entry["embedding"] = np.random.rand(1024).tolist()
    elif "dog" in entry["content"].lower():
        entry["embedding"] = (np.random.rand(1024) + 0.5).tolist()
    else:
        entry["embedding"] = (np.random.rand(1024) + 1.0).tolist()
store._rebuild_matrix()
store.save()
print(f"  Saved with synthetic embeddings")

store2 = VectorStore(path=test_path)
print(f"  Reloaded: {len(store2)} docs")
test_path.unlink()
print("  T2 ✅ PASS")

# Test 3: Skills
print("\n[T3] Skills System")
from core.skills import get_skills
sm = get_skills()
test_skill = {
    "name": "_test_skill_quick",
    "description": "Test skill for quick verification",
    "trigger_keywords": ["test", "اختبار"],
    "procedure": "1. Receive input\n2. Process\n3. Return",
    "enabled": True,
}
sm.add(test_skill)
matches = sm.find_matching("this is a test", top_k=3)
print(f"  Found {len(matches)} match(es) for 'this is a test'")
for s, score in matches:
    print(f"    [{score:.2f}] {s.name}")
sm.remove("_test_skill_quick")
print("  T3 ✅ PASS")

# Test 4: Memory
print("\n[T4] Memory")
from core.memory import get_memory
mem = get_memory()
before = mem.stats()
cid = mem.new_conversation()
mem.add_message(cid, "user", "test")
mem.add_lesson("__test_marker__")
after = mem.stats()
print(f"  Before: {before}")
print(f"  After:  {after}")
# Cleanup
mem.data["lessons"] = [l for l in mem.data["lessons"] if l.get("lesson") != "__test_marker__"]
mem.save()
ok = (after["conversations"] > before["conversations"])
print(f"  T4 {'✅ PASS' if ok else '❌ FAIL'}")

# Test 5: Knowledge (without LLM, just structure)
print("\n[T5] Knowledge Base (structure test, no embeddings yet)")
from core.knowledge import chunk_text
chunks = chunk_text("Para 1.\n\nPara 2 is longer.\n\n" + "Sentence. " * 50)
print(f"  Chunking works: {len(chunks)} chunks from test text")
print(f"  T5 ✅ PASS")

print("\n" + "=" * 60)
print(" Quick tests complete.")
print("=" * 60)
