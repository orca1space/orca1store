"""Test multilingual behavior of Hermes after language injection."""
import sys
import time
from pathlib import Path
HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

from core.orchestrator import get_orchestrator

orch = get_orchestrator()

# Warm up LLM
print("[*] Warming up LLM...")
t0 = time.time()
_ = orch.chat("ping", stream=False)
print(f"  Warmup: {time.time()-t0:.1f}s\n")

# Test cases for each language
tests = [
    ("Arabic", "مرحبا، كيف حالك؟ حدثني عن نفسك.", "العربية"),
    ("English", "Hello! How are you? Tell me about yourself.", "English"),
    ("French", "Bonjour! Comment ça va? Parle-moi de toi.", "Français"),
    ("Spanish", "¡Hola! ¿Cómo estás? Cuéntame sobre ti.", "Español"),
    ("German", "Hallo! Wie geht es dir? Erzähl mir von dir.", "Deutsch"),
]

print("=" * 60)
print(" Multilingual Test (5 languages)")
print("=" * 60)

for lang_name, msg, expected_script in tests:
    print(f"\n[{lang_name}] Test: {msg}")
    t0 = time.time()
    response = orch.chat(msg, stream=False)
    latency = time.time() - t0

    # Check if response contains the expected script characters
    has_arabic = any('\u0600' <= c <= '\u06FF' for c in response)
    has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in response)
    # For Latin scripts, harder to distinguish - check for common words
    has_french_words = any(w in response.lower() for w in ['je', 'tu', 'le', 'la', 'de', 'et', 'merci', 'bonjour'])
    has_spanish_words = any(w in response.lower() for w in ['hola', 'estoy', 'gracias', 'yo', 'tú', 'soy'])
    has_german_words = any(w in response.lower() for w in ['ich', 'bin', 'der', 'die', 'das', 'und', 'danke', 'hallo'])
    has_english_words = any(w in response.lower() for w in ['i am', 'hello', 'thanks', 'doing well', 'my name'])

    print(f"[{lang_name}] Response ({latency:.1f}s): {response[:200]}{'...' if len(response) > 200 else ''}")

    score = ""
    if expected_script == "العربية" and has_arabic: score = "✅ Arabic detected"
    elif expected_script == "English" and has_english_words: score = "✅ English words found"
    elif expected_script == "Français" and has_french_words: score = "✅ French words found"
    elif expected_script == "Español" and has_spanish_words: score = "✅ Spanish words found"
    elif expected_script == "Deutsch" and has_german_words: score = "✅ German words found"
    else: score = f"⚠️ Script detection uncertain"
    print(f"[{lang_name}] {score}")

print("\n" + "=" * 60)
print(" Status")
print("=" * 60)
status = orch.status()
print(f"  Skills: {status['skills']['count']}")
print(f"  KB chunks: {status['knowledge_base']['count']}")
print(f"  Lessons: {status['memory']['lessons']}")
