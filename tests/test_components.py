"""
Hermes Component Tests
Run each test individually to verify each layer of the system.

Usage:
    python -m tests.test_components
    python -m tests.test_components --test llm
    python -m tests.test_components --test all
"""
import sys
import json
import time
from pathlib import Path

HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

from core.config import ensure_dirs, HERMES_MODEL_FILE


def test_llm_health():
    """M1: LLM is reachable and model is loaded."""
    print("\n" + "=" * 60)
    print("TEST M1: LLM Health Check")
    print("=" * 60)
    from core.llm import get_llm
    llm = get_llm()
    print(f"  Model: {llm.model}")
    print(f"  Ollama URL: {llm.base_url}")
    models = llm.list_models()
    print(f"  Available models: {models}")
    healthy = llm.health()
    print(f"  Health: {'✅ PASS' if healthy else '❌ FAIL'}")
    return healthy


def test_llm_chat():
    """M1.2: LLM can do chat completion."""
    print("\n" + "=" * 60)
    print("TEST M1.2: LLM Chat")
    print("=" * 60)
    from core.llm import get_llm
    llm = get_llm()
    if not llm.health():
        print("  ❌ SKIP: LLM not ready")
        return False
    messages = [
        {"role": "system", "content": "You are Hermes. Reply in one short sentence."},
        {"role": "user", "content": "Say 'hello from hermes' in exactly those words."}
    ]
    result = llm.chat(messages)
    content = result.get("message", {}).get("content", "")
    print(f"  Response: {content}")
    ok = bool(content) and "hello" in content.lower()
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_llm_embed():
    """M1.3: LLM can produce embeddings."""
    print("\n" + "=" * 60)
    print("TEST M1.3: LLM Embeddings")
    print("=" * 60)
    from core.llm import get_llm
    llm = get_llm()
    if not llm.health():
        print("  ❌ SKIP: LLM not ready")
        return False
    emb1 = llm.embed("Hello world")
    emb2 = llm.embed("Goodbye world")
    emb3 = llm.embed("Bonjour le monde")
    print(f"  Vector dim: {len(emb1)}")
    print(f"  First 5 values of 'Hello world': {emb1[:5]}")
    # Quick sanity check: similar texts should have higher similarity
    import numpy as np
    def cos(a, b):
        a, b = np.array(a), np.array(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    s12 = cos(emb1, emb2)
    s13 = cos(emb1, emb3)
    print(f"  sim(Hello, Goodbye)={s12:.3f}, sim(Hello, Bonjour)={s13:.3f}")
    ok = len(emb1) > 0
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_vector_store():
    """M2: Vector store can add, search, persist."""
    print("\n" + "=" * 60)
    print("TEST M2: Vector Store")
    print("=" * 60)
    from core.vector_store import VectorStore
    from pathlib import Path
    test_path = Path("tests/test_vector_store.json")
    if test_path.exists():
        test_path.unlink()
    store = VectorStore(path=test_path)
    print(f"  Initial: {store.stats()}")
    # Add some docs
    docs = [
        "The cat sat on the mat.",
        "Dogs are loyal companions.",
        "Python is a programming language.",
        "Cats love to sleep in the sun.",
        "Machine learning models need data.",
    ]
    ids = store.add_batch(docs)
    print(f"  Added {len(ids)} docs")
    print(f"  After add: {store.stats()}")
    # Search
    results = store.search("feline pets", top_k=3)
    print(f"  Search 'feline pets':")
    for entry, score in results:
        print(f"    [{score:.3f}] {entry['content']}")
    # Save and reload
    store.save()
    store2 = VectorStore(path=test_path)
    print(f"  After reload: {store2.stats()}")
    ok = len(store2) == 5 and len(results) > 0
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    test_path.unlink()  # cleanup
    return ok


def test_knowledge():
    """M2.2: Knowledge base ingestion + search end-to-end."""
    print("\n" + "=" * 60)
    print("TEST M2.2: Knowledge Base")
    print("=" * 60)
    from core.knowledge import get_kb
    kb = get_kb()
    # Add some text
    test_text = """
    Hermes is a local AI agent. It runs entirely on the user's machine.
    It uses Ollama to run open-source language models like Qwen 2.5.
    Knowledge is stored in a vector database for semantic search.
    The user teaches Hermes through skills, lessons, and documents.
    """
    ids = kb.add_text(test_text, source="test")
    print(f"  Ingested test text: {len(ids)} chunks")
    results = kb.search("Where does Hermes run?", top_k=3)
    print(f"  Search 'Where does Hermes run?':")
    for r in results:
        print(f"    [{r['score']:.3f}] {r['content'][:80]}...")
    ok = len(results) > 0
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_skills():
    """M3: Skills system can register, match, and use skills."""
    print("\n" + "=" * 60)
    print("TEST M3: Skills")
    print("=" * 60)
    from core.skills import get_skills
    sm = get_skills()
    # Add a test skill
    test_skill = {
        "name": "test_greet",
        "description": "Greets the user warmly in Arabic",
        "trigger_keywords": ["greet", "سلام", "مرحبا", "hello"],
        "procedure": "1. Receive user's name if provided\n2. Compose a warm greeting in Arabic\n3. If user provided a name, use it",
        "examples": [{"input": "مرحبا يا أحمد", "output": "أهلاً وسهلاً يا أحمد، نورت!"}],
        "enabled": True,
    }
    sm.add(test_skill)
    print(f"  Added 'test_greet' skill")
    # Match
    matches = sm.find_matching("مرحبا كيف حالك؟", top_k=3)
    print(f"  Matching for 'مرحبا كيف حالك؟':")
    for s, score in matches:
        print(f"    [{score:.2f}] {s.name}: {s.description}")
    # Cleanup
    sm.remove("test_greet")
    ok = len(matches) > 0
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_memory():
    """Memory persistence test."""
    print("\n" + "=" * 60)
    print("TEST: Memory")
    print("=" * 60)
    from core.memory import get_memory
    mem = get_memory()
    before = mem.stats()
    conv_id = mem.new_conversation()
    mem.add_message(conv_id, "user", "test message")
    mem.add_lesson("Test lesson for verification")
    after = mem.stats()
    print(f"  Before: {before}")
    print(f"  After:  {after}")
    ok = (after["conversations"] > before["conversations"] and
          after["lessons"] > before["lessons"])
    # Cleanup
    mem.data["lessons"] = [l for l in mem.data["lessons"] if l["lesson"] != "Test lesson for verification"]
    mem.save()
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_orchestrator():
    """M4: End-to-end orchestrator test."""
    print("\n" + "=" * 60)
    print("TEST M4: Orchestrator (end-to-end)")
    print("=" * 60)
    from core.orchestrator import get_orchestrator
    orch = get_orchestrator()
    status = orch.status()
    print(f"  LLM: {status['llm']['model']}, healthy={status['llm']['healthy']}")
    print(f"  KB chunks: {status['knowledge_base']['count']}")
    print(f"  Skills: {status['skills']['count']}")
    if not status["llm"]["healthy"]:
        print("  ❌ SKIP: LLM not ready")
        return False
    # First teach something specific
    orch.teach_lesson("Always answer in Arabic when the user writes in Arabic.")
    orch.teach_document(
        "Hermes is the Greek messenger god, swift and clever. "
        "In our system, Hermes is a local AI agent that learns only from its user.",
        source="training"
    )
    # Test chat
    response = orch.chat("من هو هيرمس؟", stream=False)
    print(f"  Q: من هو هيرمس؟")
    print(f"  A: {response[:300]}...")
    has_arabic = any('\u0600' <= c <= '\u06FF' for c in response)
    has_relevant = "وكيل" in response or "محلي" in response or "hermes" in response.lower() or "هيرمس" in response
    ok = bool(response) and (has_arabic or has_relevant)
    print(f"  Arabic chars in response: {has_arabic}")
    print(f"  Mentions relevant concept: {has_relevant}")
    print(f"  Result: {'✅ PASS' if ok else '⚠️  PARTIAL (LLM not following lesson yet)'}")
    return ok


def run_all():
    ensure_dirs()
    print("=" * 60)
    print(" HERMES — Test Suite")
    print("=" * 60)
    results = {}
    for name, fn in [
        ("M1.1 LLM Health", test_llm_health),
        ("M1.2 LLM Chat", test_llm_chat),
        ("M1.3 LLM Embed", test_llm_embed),
        ("M2.1 Vector Store", test_vector_store),
        ("M2.2 Knowledge Base", test_knowledge),
        ("M3 Skills", test_skills),
        ("Memory", test_memory),
        ("M4 Orchestrator", test_orchestrator),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  Total: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["all", "llm", "llm_chat", "llm_embed",
                                 "vector", "knowledge", "skills",
                                 "memory", "orchestrator"])
    args = parser.parse_args()
    if args.test == "all":
        run_all()
    elif args.test == "llm":
        test_llm_health()
    elif args.test == "llm_chat":
        test_llm_chat()
    elif args.test == "llm_embed":
        test_llm_embed()
    elif args.test == "vector":
        test_vector_store()
    elif args.test == "knowledge":
        test_knowledge()
    elif args.test == "skills":
        test_skills()
    elif args.test == "memory":
        test_memory()
    elif args.test == "orchestrator":
        test_orchestrator()
