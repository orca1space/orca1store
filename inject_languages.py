"""
Language Pack Injector
Teaches Hermes a set of major world languages with skills, lessons, and conversation patterns.

Each language gets:
- A conversation skill (greeting, common phrases, cultural context)
- A translation skill
- A grammar assistant skill
- Lessons about the language and how to respond in it

Default languages: Arabic (primary), English (primary), French, Spanish, German
"""
import sys
from pathlib import Path
HERMES_ROOT = Path(__file__).parent
sys.path.insert(0, str(HERMES_ROOT))

from core.orchestrator import get_orchestrator
from core.skills import get_skills
from core.knowledge import get_kb
import json


# ============================================================
# Language Packs
# ============================================================
LANGUAGE_PACKS = {
    "arabic": {
        "name_native": "العربية",
        "name_english": "Arabic",
        "code": "ar",
        "direction": "rtl",
        "greetings": [
            "السلام عليكم", "أهلاً وسهلاً", "صباح الخير", "مساء الخير",
            "مرحبا", "أهلا", "يا هلا", "حياك الله", "نورت"
        ],
        "common_phrases": {
            "how_are_you": "كيف حالك؟ / إيش الأخبار؟",
            "thank_you": "شكراً / شكراً جزيلاً / الله يعطيك العافية",
            "please": "لو سمحت / من فضلك",
            "yes_no": "نعم / لا / أيوه / لأ",
            "goodbye": "مع السلامة / في أمان الله / إلى اللقاء",
            "sorry": "آسف / عذراً / المعذرة",
            "i_understand": "فاهم / مفهوم / تمام",
            "i_dont_understand": "مش فاهم / ما فهمت / وضح لي",
        },
        "cultural_notes": (
            "Arabic is the primary language for the user. Respond in Arabic by default when the user writes in Arabic. "
            "Use Modern Standard Arabic (فصحى) for formal content, but you can use natural conversational Egyptian/Levantine "
            "dialects for casual chat. The user appreciates warm, friendly tone. Use appropriate honorifics when relevant."
        ),
    },
    "english": {
        "name_native": "English",
        "name_english": "English",
        "code": "en",
        "direction": "ltr",
        "greetings": [
            "Hello", "Hi", "Hey", "Good morning", "Good afternoon",
            "Good evening", "Greetings", "Welcome", "Howdy", "Hi there"
        ],
        "common_phrases": {
            "how_are_you": "How are you? / How's it going? / What's up?",
            "thank_you": "Thank you / Thanks / Thanks a lot / Much appreciated",
            "please": "Please / If you don't mind",
            "yes_no": "Yes / No / Yeah / Nope / Sure",
            "goodbye": "Goodbye / Bye / See you / Take care / Farewell",
            "sorry": "Sorry / My apologies / Pardon me",
            "i_understand": "I understand / Got it / Understood / Makes sense",
            "i_dont_understand": "I don't understand / Could you clarify? / Please explain",
        },
        "cultural_notes": (
            "English is the secondary language for the user. Respond in English when the user writes in English. "
            "Be clear, concise, and friendly. Use natural conversational English, not overly formal. "
            "Use contractions (I'm, you're, don't) for natural flow unless the context is very formal."
        ),
    },
    "french": {
        "name_native": "Français",
        "name_english": "French",
        "code": "fr",
        "direction": "ltr",
        "greetings": [
            "Bonjour", "Salut", "Bonsoir", "Bonne journée",
            "Bienvenue", "Coucou", "Allô", "Enchanté"
        ],
        "common_phrases": {
            "how_are_you": "Comment ça va ? / Comment allez-vous ? / Ça va ?",
            "thank_you": "Merci / Merci beaucoup / Je vous remercie",
            "please": "S'il vous plaît / S'il te plaît",
            "yes_no": "Oui / Non / Bien sûr",
            "goodbye": "Au revoir / À bientôt / Salut / Adieu",
            "sorry": "Désolé / Pardon / Excusez-moi / Je m'excuse",
            "i_understand": "Je comprends / D'accord / Compris",
            "i_dont_understand": "Je ne comprends pas / Pouvez-vous clarifier ?",
        },
        "cultural_notes": (
            "French is a Romance language spoken widely in Europe, Africa, and Canada. "
            "When the user writes in French, respond in French. Use 'tu' for informal and 'vous' for formal. "
            "Maintain politeness markers (bonjour, merci, s'il vous plaît) — they are essential in French culture."
        ),
    },
    "spanish": {
        "name_native": "Español",
        "name_english": "Spanish",
        "code": "es",
        "direction": "ltr",
        "greetings": [
            "Hola", "Buenos días", "Buenas tardes", "Buenas noches",
            "Qué tal", "Bienvenido", "Cómo estás"
        ],
        "common_phrases": {
            "how_are_you": "¿Cómo estás? / ¿Qué tal? / ¿Cómo te va?",
            "thank_you": "Gracias / Muchas gracias / Te agradezco",
            "please": "Por favor",
            "yes_no": "Sí / No / Claro / Por supuesto",
            "goodbye": "Adiós / Hasta luego / Nos vemos / Chao",
            "sorry": "Lo siento / Perdón / Disculpa",
            "i_understand": "Entiendo / De acuerdo / Comprendido",
            "i_dont_understand": "No entiendo / ¿Puedes aclarar?",
        },
        "cultural_notes": (
            "Spanish is spoken in Spain, Latin America, and parts of the US. "
            "When the user writes in Spanish, respond in Spanish. "
            "Be warm and friendly — Spanish-speaking cultures value personal connection. "
            "Use 'tú' for informal and 'usted' for formal. The user may use either dialect."
        ),
    },
    "german": {
        "name_native": "Deutsch",
        "name_english": "German",
        "code": "de",
        "direction": "ltr",
        "greetings": [
            "Hallo", "Guten Morgen", "Guten Tag", "Guten Abend",
            "Servus", "Moin", "Grüß Gott", "Willkommen"
        ],
        "common_phrases": {
            "how_are_you": "Wie geht es dir? / Wie geht's? / Was geht?",
            "thank_you": "Danke / Vielen Dank / Danke schön / Danke sehr",
            "please": "Bitte",
            "yes_no": "Ja / Nein / Klar / Sicher",
            "goodbye": "Auf Wiedersehen / Tschüss / Bis später / Ciao",
            "sorry": "Entschuldigung / Es tut mir leid / Sorry",
            "i_understand": "Ich verstehe / Verstanden / Klar",
            "i_dont_understand": "Ich verstehe nicht / Kannst du das erklären?",
        },
        "cultural_notes": (
            "German is spoken in Germany, Austria, and Switzerland. "
            "When the user writes in German, respond in German. "
            "Be direct and precise — German speakers value clarity. "
            "Use 'du' for informal and 'Sie' for formal. Maintain proper capitalization of nouns."
        ),
    },
}


def inject_language_skills(orch):
    """Add skills for each language."""
    skills_mgr = orch.skills
    added = 0

    for lang_key, pack in LANGUAGE_PACKS.items():
        # 1. Conversation skill
        conv_skill = {
            "name": f"converse_{lang_key}",
            "description": f"Converse naturally in {pack['name_english']} ({pack['name_native']})",
            "trigger_keywords": pack["greetings"][:5] + [pack["name_native"], pack["name_english"].lower()],
            "procedure": (
                f"1. Detect that the user is writing in {pack['name_english']} or wants to speak it.\n"
                f"2. Respond in {pack['name_native']} ({pack['name_english']}).\n"
                f"3. Match the formality level (formal vs informal) of the user.\n"
                f"4. Use natural, conversational {pack['name_english']} — avoid overly stiff or textbook phrasing.\n"
                f"5. Cultural note: {pack['cultural_notes']}"
            ),
            "examples": [
                {"input": pack["greetings"][0], "output": f"{pack['greetings'][1]}!"},
                {"input": pack["common_phrases"]["how_are_you"], "output": "I'm doing well, thanks for asking!"},
            ],
            "enabled": True,
        }
        skills_mgr.add(conv_skill)
        added += 1

        # 2. Translation skill
        trans_skill = {
            "name": f"translate_{lang_key}",
            "description": f"Translate text to or from {pack['name_english']} ({pack['name_native']})",
            "trigger_keywords": [
                f"translate to {pack['name_english']}", f"ترجم إلى {pack['name_native']}",
                f"in {pack['name_english']}", f"بـ{pack['name_native']}",
                f"translate {pack['name_english']}", f"ترجمة {pack['name_native']}"
            ],
            "procedure": (
                f"1. Identify the source language and target language.\n"
                f"2. If translating TO {pack['name_english']}: produce a natural, idiomatic {pack['name_english']} translation.\n"
                f"3. If translating FROM {pack['name_english']}: detect target language and translate accurately.\n"
                f"4. Preserve meaning, tone, and cultural nuances.\n"
                f"5. If a phrase has no direct equivalent, provide the closest cultural match with a brief note."
            ),
            "examples": [
                {"input": f"Translate 'hello' to {pack['name_english']}", "output": pack["greetings"][0]},
            ],
            "enabled": True,
        }
        skills_mgr.add(trans_skill)
        added += 1

        # 3. Common phrases skill
        phrases_skill = {
            "name": f"phrases_{lang_key}",
            "description": f"Common everyday phrases in {pack['name_english']}",
            "trigger_keywords": [
                f"common phrases in {pack['name_english']}",
                f"how to say in {pack['name_english']}",
                f"عبارات {pack['name_native']}",
                f"كيف أقول بـ{pack['name_native']}",
            ],
            "procedure": (
                f"Provide common {pack['name_english']} phrases for daily use:\n"
                + "\n".join([f"- {k}: {v}" for k, v in pack["common_phrases"].items()])
            ),
            "examples": [],
            "enabled": True,
        }
        skills_mgr.add(phrases_skill)
        added += 1

    return added


def inject_language_lessons(orch):
    """Add language-related lessons to the system prompt."""
    lessons = [
        "ALWAYS respond in the language the user is writing in. If they write in Arabic, respond in Arabic. If English, respond in English. Match their language exactly.",
        "When the user mixes languages (code-switching), follow the dominant language of the current message but stay flexible.",
        "Maintain the cultural tone appropriate to each language: warm for Arabic, friendly for English, polite for French, warm for Spanish, direct for German.",
        "Never use machine-translation feel. Sound like a native speaker with natural idioms and contractions where appropriate.",
        "If the user asks for translation, provide natural idiomatic translation, not literal word-for-word.",
    ]
    added = 0
    for lesson in lessons:
        orch.teach_lesson(lesson)
        added += 1
    return added


def inject_language_knowledge(orch):
    """Add reference data about each language to the knowledge base."""
    chunks_added = 0
    for lang_key, pack in LANGUAGE_PACKS.items():
        # Main language description
        desc = (
            f"# {pack['name_english']} ({pack['name_native']})\n\n"
            f"Language code: {pack['code']}\n"
            f"Script direction: {pack['direction']}\n\n"
            f"## Greetings\n"
            + "\n".join([f"- {g}" for g in pack["greetings"]])
            + f"\n\n## Common Phrases\n"
            + "\n".join([f"- {k}: {v}" for k, v in pack["common_phrases"].items()])
            + f"\n\n## Cultural Notes\n{pack['cultural_notes']}"
        )
        ids = orch.teach_document(desc, source=f"language_pack:{lang_key}")
        chunks_added += len(ids)
    return chunks_added


def main():
    print("=" * 60)
    print(" Hermes — Language Pack Injector")
    print("=" * 60)

    orch = get_orchestrator()

    print(f"\n[1] Adding skills for {len(LANGUAGE_PACKS)} languages...")
    skills_added = inject_language_skills(orch)
    print(f"    Added {skills_added} skills")

    print(f"\n[2] Adding language lessons...")
    lessons_added = inject_language_lessons(orch)
    print(f"    Added {lessons_added} lessons")

    print(f"\n[3] Adding language knowledge base entries...")
    chunks_added = inject_language_knowledge(orch)
    print(f"    Added {chunks_added} knowledge chunks")

    print(f"\n[4] Saving...")
    orch.kb.save()

    print(f"\n[5] Final status:")
    status = orch.status()
    print(f"    Total skills: {status['skills']['count']}")
    print(f"    Total lessons: {status['memory']['lessons']}")
    print(f"    KB chunks: {status['knowledge_base']['count']}")

    print("\n" + "=" * 60)
    print(" Language injection complete!")
    print("=" * 60)
    print(f"\nLanguages installed: {', '.join(p['name_english'] for p in LANGUAGE_PACKS.values())}")


if __name__ == "__main__":
    main()
