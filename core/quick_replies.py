"""
Hermes Quick Replies
Instant, no-LLM responses for common patterns:
- Greetings (in all 5 languages)
- Time queries
- Identity questions about Hermes itself
- Thanks / acknowledgments
- Goodbye

These bypass the LLM entirely. Saves 5-15 seconds per query.
Pure Python. No external dependencies. Fully self-contained.
"""
import re
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple


# Each rule: (pattern, response, language)
# Patterns are matched case-insensitively, whitespace-normalized.
QUICK_RULES: List[Tuple[str, str, str]] = [
    # === Arabic greetings ===
    (r"^(السلام عليكم|سلام عليكم|سلام)$",
     "وعليكم السلام ورحمة الله وبركاته. كيف أقدر أساعدك؟", "ar"),
    (r"^(صباح الخير|صباح النور)$",
     "صباح النور! يومك سعيد. شو اللي تحب نبدأ بيه؟", "ar"),
    (r"^(مساء الخير|مساء النور|مساء الورد)$",
     "مساء النور! أهلاً وسهلاً. كيف أقدر أساعدك؟", "ar"),
    (r"^(أهلا|أهلا وسهلا|اهلا|اهلا وسهلا|مرحبا|مرحبًا|هاي|هلاء)$",
     "أهلاً وسهلاً فيك! جاهز أخدمك في أي شي.", "ar"),
    (r"^(كيف حالك|كيفك|كيفك انت|كيف الحال|شو اخبارك)$",
     "الحمد لله تمام، وجاهز أساعدك. إنت كيفك؟", "ar"),
    (r"^(شكرا|شكراً|شكرا لك|شكرا جزيلا|شكرًا جزيلًا|يسلمو|يسلمو ايديك)$",
     "العفو، تحت أمرك في أي وقت.", "ar"),
    (r"^(وداعا|وداعًا|الى اللقاء|مع السلامة|باي)$",
     "في أمان الله. أكون هون لما تحتاجني.", "ar"),

    # === English greetings ===
    (r"^(hello|hi|hey|hi there|hello there)$",
     "Hi! How can I help you today?", "en"),
    (r"^(good morning|morning)$",
     "Good morning! Hope your day is going well. What's on your mind?", "en"),
    (r"^(good evening|good afternoon|evening)$",
     "Good evening! How can I help you?", "en"),
    (r"^(how are you|how are you doing|how's it going|sup|what's up)$",
     "I'm doing great, ready to help. How are you?", "en"),
    (r"^(thanks|thank you|thx|ty|thank you so much|cheers)$",
     "You're welcome! Always here to help.", "en"),
    (r"^(bye|goodbye|see you|see ya|cya|farewell)$",
     "Goodbye! I'll be here whenever you need me.", "en"),

    # === French greetings ===
    (r"^(bonjour|salut|bonsoir|coucou)$",
     "Bonjour ! Comment puis-je vous aider ?", "fr"),
    (r"^(bonne journée|bonne soirée)$",
     "À vous aussi ! Comment puis-je vous aider ?", "fr"),
    (r"^(comment ça va|comment allez-vous|ça va)$",
     "Je vais bien, merci. Et vous ?", "fr"),
    (r"^(merci|merci beaucoup|je vous remercie)$",
     "Avec plaisir ! N'hésitez pas si vous avez besoin.", "fr"),
    (r"^(au revoir|à bientôt|adieu|ciao)$",
     "Au revoir ! Je serai là quand vous aurez besoin.", "fr"),

    # === German greetings ===
    (r"^(hallo|hi|hey|guten tag|tag|servus|grüß dich)$",
     "Hallo! Wie kann ich Ihnen helfen?", "de"),
    (r"^(guten morgen|morgen)$",
     "Guten Morgen! Wie kann ich behilflich sein?", "de"),
    (r"^(guten abend|abend)$",
     "Guten Abend! Was kann ich für Sie tun?", "de"),
    (r"^(wie geht es dir|wie geht's|wie geht es ihnen)$",
     "Mir geht es gut, danke. Und Ihnen?", "de"),
    (r"^(danke|dankeschön|danke schön|vielen dank)$",
     "Gerne! Immer zur Verfügung.", "de"),
    (r"^(tschüss|auf wiedersehen|bis bald|ciao)$",
     "Auf Wiedersehen! Ich bin hier, wenn Sie mich brauchen.", "de"),

    # === Spanish greetings ===
    (r"^(hola|buenos días|buenas tardes|buenas noches|qué tal)$",
     "¡Hola! ¿En qué puedo ayudarte?", "es"),
    (r"^(buenos dias|buenas tardes|buenas noches)$",
     "¡Buenas! ¿Cómo puedo asistirte?", "es"),
    (r"^(cómo estás|como estas|qué tal estás|que tal estas)$",
     "¡Muy bien, gracias! ¿Y tú?", "es"),
    (r"^(gracias|muchas gracias|mil gracias)$",
     "¡De nada! Estoy aquí para ayudarte.", "es"),
    (r"^(adiós|adios|hasta luego|chao|chau|hasta pronto)$",
     "¡Adiós! Aquí estaré cuando me necesites.", "es"),

    # === Identity / meta questions ===
    (r"^(من انت|من أنت|ما اسمك|انت مين|مين انت|عرفني بنفسك)$",
     "أنا Hermes، مساعد ذكاء اصطناعي محلي شغّال على جهازك. مصنوع خصيصاً ليخدمك إنت فقط.", "ar"),
    (r"^(who are you|what are you|what's your name|tell me about yourself|introduce yourself)$",
     "I'm Hermes, a local AI assistant running on your machine. I'm here to serve you — the user is my sole authority.", "en"),
    (r"^(qui es-tu|qui êtes-vous|comment tu t'appelles)$",
     "Je suis Hermes, un assistant IA local qui fonctionne sur votre machine. Je suis à votre service.", "fr"),
    (r"^(wer bist du|wie heißt du|stell dich vor)$",
     "Ich bin Hermes, ein lokaler KI-Assistent auf Ihrem Computer. Ich bin zu Ihren Diensten.", "de"),
    (r"^(quién eres|quien eres|cómo te llamas)$",
     "Soy Hermes, un asistente de IA local en tu máquina. Estoy a tu servicio.", "es"),

    # === Time queries ===
    (r"^(كم الساعة|كم الساعه|الوقت|الساعة كام|what time is it|what's the time|quelle heure|wie spät|qué hora)$",
     "__TIME__", "any"),
    (r"^(اليوم كام|اي يوم|what day|date today|aujourd'hui|welcher tag|qué día)$",
     "__DATE__", "any"),
]


def _norm(text: str) -> str:
    """Normalize text for matching: lowercase, strip, collapse whitespace, remove trailing punctuation."""
    t = " ".join(text.lower().strip().split())
    # Strip trailing punctuation: . ! ? ، : ; ۔ ٪
    while t and t[-1] in ".!?،:;؟۔%":
        t = t[:-1].rstrip()
    return t


def try_quick_reply(user_message: str) -> Optional[str]:
    """
    If the message matches a quick-reply pattern, return the canned response.
    Returns None if no match (caller should fall through to LLM).
    """
    if not user_message:
        return None
    msg = _norm(user_message)
    for pattern, response, lang in QUICK_RULES:
        try:
            if re.match(pattern, msg):
                if response == "__TIME__":
                    now = datetime.now()
                    return f"الساعة الآن {now.strftime('%H:%M')} ({now.strftime('%Y-%m-%d')})."
                if response == "__DATE__":
                    now = datetime.now()
                    days = ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
                    return f"اليوم {days[now.weekday()]}، {now.strftime('%Y-%m-%d')}."
                return response
        except re.error:
            continue
    return None


def stats() -> Dict:
    return {"rules": len(QUICK_RULES)}


if __name__ == "__main__":
    test = ["مرحبا", "hello", "السلام عليكم", "thanks", "من أنت",
            "كم الساعة", "weather in cairo", "goodbye"]
    for t in test:
        r = try_quick_reply(t)
        print(f"  {t!r:30} -> {r}")
