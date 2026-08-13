"""Final smoke test: simulate the CLI chat experience."""
import sys
import time
from pathlib import Path
HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

print("=" * 60)
print(" HERMES — Final Smoke Test")
print("=" * 60)

from core.orchestrator import get_orchestrator

orch = get_orchestrator()

# Pre-teach a personality
orch.teach_lesson("Always respond in the language the user writes in.")
orch.teach_lesson("You are Hermes, a local AI assistant. You learn only from your user.")
orch.teach_lesson("If you don't know something, say so honestly.")

# Pre-teach a skill
orch.teach_skill({
    "name": "time_greeting",
    "description": "Greets the user with a time-appropriate greeting",
    "trigger_keywords": ["صباح", "مساء", "morning", "evening", "hello", "مرحبا"],
    "procedure": "1. Detect language\n2. Compose a warm, time-aware greeting",
    "enabled": True,
})

# Pre-load a small knowledge snippet
orch.teach_document(
    "Hermes can speak Arabic and English fluently. It uses Qwen 2.5 3B model "
    "running locally via llama-cpp-python. It was created in 2026.",
    source="intro"
)

# Pre-warm the LLM
print("\n[*] Warming up LLM (first call loads model)...")
t0 = time.time()
_ = orch.chat("hi", stream=False)
print(f"  LLM warmup: {time.time()-t0:.1f}s")

# Now simulate a short chat session
print("\n[*] Simulating 3-turn conversation:")
print("-" * 60)

test_turns = [
    ("User", "مرحبا هيرمس، من أنت؟"),
    ("User", "What can you do?"),
    ("User", "ما النموذج الذي تستخدمه؟"),
]

for role, msg in test_turns:
    print(f"\n[{role}]: {msg}")
    t0 = time.time()
    response = orch.chat(msg, stream=False)
    latency = time.time() - t0
    print(f"[Hermes] ({latency:.1f}s): {response}")

# Show final state
print("\n" + "=" * 60)
print(" FINAL STATUS")
print("=" * 60)
import json
status = orch.status()
print(json.dumps(status, indent=2, ensure_ascii=False, default=str))

print("\n" + "=" * 60)
print(" ✅ Hermes is operational")
print("=" * 60)
print("\nTo start chatting interactively, run:")
print("  python cli.py chat")
print("\nTo teach Hermes, run:")
print("  python train.py")
print("\nTo run all tests, run:")
print("  python -m tests.test_components")
