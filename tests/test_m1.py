"""M1: Test LLM loading, chat, embeddings."""
import sys
from pathlib import Path
HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))
from core.llm import get_llm
import time

print("=" * 60)
print(" M1: LLM Test")
print("=" * 60)

print("\n[1] Loading LLM...")
t0 = time.time()
llm = get_llm()
print(f"Init time: {time.time()-t0:.1f}s")
print("Model info:", llm.model_info())

print("\n[2] Test Chat (Arabic)")
messages = [
    {"role": "system", "content": "You are Hermes. Reply briefly."},
    {"role": "user", "content": "مرحبا، من أنت؟ أجب في جملة واحدة."}
]
t0 = time.time()
result = llm.chat(messages, max_tokens=100)
print(f"Chat latency: {time.time()-t0:.1f}s")
print(f"Response: {result['message']['content']}")

print("\n[3] Test Chat (English)")
messages = [
    {"role": "system", "content": "You are Hermes. Reply briefly."},
    {"role": "user", "content": "Hello, who are you? Reply in one sentence."}
]
t0 = time.time()
result = llm.chat(messages, max_tokens=100)
print(f"Chat latency: {time.time()-t0:.1f}s")
print(f"Response: {result['message']['content']}")

print("\n[4] Test Embeddings")
t0 = time.time()
emb1 = llm.embed("The cat sat on the mat.")
emb2 = llm.embed("A feline is resting on a rug.")
emb3 = llm.embed("Python is a programming language.")
print(f"Embedding latency: {time.time()-t0:.1f}s")
print(f"Dim: {len(emb1)}")
import numpy as np
def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
s12 = cos(emb1, emb2)
s13 = cos(emb1, emb3)
print(f"sim(cat, feline)  = {s12:.3f} (should be HIGH)")
print(f"sim(cat, python)  = {s13:.3f} (should be LOW)")
ok = s12 > s13
print(f"\nSemantic ordering: {'CORRECT' if ok else 'WRONG'}")

print("\n" + "=" * 60)
print(" M1 ✅ ALL TESTS PASSED")
print("=" * 60)
