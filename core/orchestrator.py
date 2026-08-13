"""
Hermes Orchestrator
The brain of Hermes. Coordinates LLM, knowledge base, skills, and memory.

Flow for each user message:
1. Search knowledge base for relevant context
2. Find matching skills
3. Build a system prompt with: base identity + relevant knowledge + matching skills
4. Build message history with current conversation
5. Call LLM
6. Persist exchange to memory
7. Return response

Pure Python. No external frameworks.
"""
import json
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import (
    HERMES_BASE_PROMPT, LLM_TEMPERATURE,
    TOP_K_RESULTS, SIMILARITY_THRESHOLD,
)
from core.llm import get_llm, HermesLLM
from core.knowledge import get_kb
from core.skills import get_skills
from core.memory import get_memory
from core.cache import get_cache
from core.quick_replies import try_quick_reply
from core.session_state import get_session
from core.skill_search import get_skill_search
from core.skill_library import get_library
from core.checkpoint import get_checkpoint_manager
from core.time_travel import get_time_travel
from core.logging_setup import get_logger

log = get_logger("orchestrator")
from core.multi_agent import get_multi_agent
from core.human_in_loop import get_hitl
from core.memory_store import get_memory_store
from core.hybrid_search import get_hybrid_search


MAX_RESTORED_MESSAGES = 50


class HermesOrchestrator:
    """The main brain that coordinates all components."""

    def __init__(self):
        self.llm: HermesLLM = get_llm()
        self.kb = get_kb()
        self.skills = get_skills()
        self.memory = get_memory()
        self.cache = get_cache()
        self.session = get_session()
        self.skill_search = get_skill_search()
        self.library = get_library()
        self.checkpoint_mgr = get_checkpoint_manager()
        self.time_travel = get_time_travel()
        self.multi_agent = get_multi_agent()
        self.hitl = get_hitl()
        self.memory_store = get_memory_store()
        self.hybrid_search = get_hybrid_search()
        # Ensure there's always at least one tab
        if not self.session.list_tabs():
            self.session.create_tab("Welcome")
        self._current_conv_id: Optional[str] = self.session.get_active_tab_id()

    def start_conversation(self) -> str:
        """Start a new conversation thread (legacy - now uses tabs)."""
        return self.session.create_tab().get("id", "")

    def restore_last_session(self) -> Dict:
        """Restore the last session if it exists. Returns full state."""
        state = self.session.restore()
        if state.get("active_tab_id"):
            self._current_conv_id = state["active_tab_id"]
        return state

    def dynamic_max_tokens(self, user_message: str) -> int:
        """Choose max_tokens based on question length. Shorter Q = shorter A = faster.

        We deliberately cap aggressively because:
        - 3B model on CPU takes ~0.4s per token
        - Most questions need <100 tokens to answer
        - If the answer needs more, the user can ask for detail
        """
        msg_len = len(user_message)
        if msg_len < 20:
            return 96        # "hi", "hello", "مرحبا"
        elif msg_len < 60:
            return 192       # short question
        elif msg_len < 200:
            return 256       # medium question
        else:
            return 384       # long question

    def select_skills_for_query(self, user_message: str, top_k: int = 2,
                                min_score: float = 0.30) -> List:
        """
        Smart skill auto-selection: uses semantic search (embeddings)
        to find skills most relevant to the user's query.
        Returns list of (skill, score) tuples.
        """
        try:
            results = self.skill_search.search(user_message, top_k=top_k, min_score=min_score)
            return [(skill, score) for _, score, skill in results]
        except Exception as e:
            print(f"[orchestrator] skill_search failed: {e}")
            # Fallback to keyword matching
            return self.skills.find_matching(user_message, top_k=top_k, min_score=0.15)

    # === TAB OPERATIONS ===

    def list_tabs(self) -> List[Dict]:
        """List all tabs (lightweight)."""
        return self.session.list_tabs()

    def get_tab(self, tab_id: str) -> Optional[Dict]:
        return self.session.get_tab(tab_id)

    def create_tab(self, title: Optional[str] = None) -> Dict:
        new_tab = self.session.create_tab(title)
        # Mirror in memory so legacy code paths work
        try:
            self.memory.new_conversation(conversation_id=new_tab["id"])
        except Exception:
            pass
        return new_tab

    def close_tab(self, tab_id: str) -> bool:
        return self.session.close_tab(tab_id)

    def switch_tab(self, tab_id: str) -> Optional[Dict]:
        tab = self.session.switch_tab(tab_id)
        if tab is not None:
            self._current_conv_id = tab_id
        return tab

    def get_active_tab(self) -> Optional[Dict]:
        return self.session.get_active_tab()

    @property
    def current_conversation(self) -> Optional[str]:
        if self._current_conv_id is None:
            # Make sure there is an active tab
            active = self.session.get_active_tab()
            self._current_conv_id = active["id"] if active else None
        return self._current_conv_id

    def _build_context(self, user_message: str) -> str:
        """
        Build the system context for this turn. Keep it small (under 1500 chars
        after truncation) to avoid exceeding n_ctx with the response.
        """
        sections = [HERMES_BASE_PROMPT]

        # 1. Knowledge retrieval — only top 2, each truncated to 400 chars
        kb_results = self.kb.search(user_message, top_k=2,
                                    threshold=SIMILARITY_THRESHOLD)
        if kb_results:
            sections.append("## Retrieved Knowledge")
            for i, r in enumerate(kb_results, 1):
                src = r["metadata"].get("filename") or r["metadata"].get("source", "unknown")
                content = r["content"][:400] + ("..." if len(r["content"]) > 400 else "")
                sections.append(f"[{i}] {src}: {content}")

        # 2. Smart skill auto-selection — only top 1
        matching_skills = self.select_skills_for_query(user_message, top_k=1)
        if matching_skills:
            skill, score = matching_skills[0]
            proc = skill.procedure[:300] + ("..." if len(skill.procedure) > 300 else "")
            sections.append(
                f"## Skill: {skill.name} (match: {score:.2f})\n{proc}"
            )

        # 3. Recent lessons (max 3, truncated)
        lessons = self.memory.get_lessons(limit=3)
        if lessons:
            sections.append("## Lessons")
            for l in lessons:
                sections.append(f"- {l['lesson'][:200]}")

        return "\n\n".join(sections)

    def chat(self, user_message: str, stream: bool = False,
             use_cache: bool = True, tab_id: Optional[str] = None) -> str:
        """
        Main entry point. Send a user message, get a response.
        Persists both sides to memory + session tab.

        Lookup order:
        1. Quick replies (greetings, identity, time) — no LLM
        2. Response cache — no LLM
        3. LLM call — stored in cache for next time
        """
        t0 = time.time()
        log = logging.getLogger("hermes.orchestrator.chat")

        if not self.llm.health():
            return ("[Hermes Error] Local LLM is not ready. "
                    "Make sure the model file exists at the configured path "
                    "and llama-cpp-python is installed.")

        if tab_id is None:
            tab_id = self.current_conversation
        if tab_id is None or self.session.get_tab(tab_id) is None:
            new_tab = self.session.create_tab()
            tab_id = new_tab["id"]
            self._current_conv_id = tab_id
        log.debug("setup: %.2fs", time.time() - t0)

        # 1. Quick replies (no LLM, no memory write of "thinking")
        qr = try_quick_reply(user_message)
        if qr is not None:
            log.debug("quick_reply: %.2fs (matched)", time.time() - t0)
            self.memory.add_message(tab_id, "user", user_message)
            log.debug("memory.add user: %.2fs", time.time() - t0)
            self.memory.add_message(tab_id, "assistant", qr)
            log.debug("memory.add asst: %.2fs", time.time() - t0)
            self.session.append_message(tab_id, "user", user_message)
            log.debug("session.add user: %.2fs", time.time() - t0)
            self.session.append_message(tab_id, "assistant", qr)
            log.debug("session.add asst: %.2fs", time.time() - t0)
            return qr

        # 2. Cache lookup (if enabled)
        t1 = time.time()
        if use_cache:
            cached = self.cache.get(user_message)
            if cached is not None:
                log.debug("cache hit: %.2fs", time.time() - t1)
                self.memory.add_message(tab_id, "user", user_message)
                self.memory.add_message(tab_id, "assistant", cached)
                self.session.append_message(tab_id, "user", user_message)
                self.session.append_message(tab_id, "assistant", cached)
                return cached

        # Persist user message
        self.memory.add_message(tab_id, "user", user_message)
        self.session.append_message(tab_id, "user", user_message)

        # Build context
        t_ctx = time.time()
        system_context = self._build_context(user_message)
        log.debug("build_context: %.2fs len=%d", time.time() - t_ctx, len(system_context))

        # Build messages: system + recent history + new user msg
        history = self.memory.get_recent_messages(tab_id, n=8)
        messages = [{"role": "system", "content": system_context}]
        for m in history[:-1]:  # exclude the just-added user message
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_message})

        # Call LLM
        t_llm = time.time()
        try:
            max_tok = self.dynamic_max_tokens(user_message)
            if stream:
                response_parts = []
                for chunk in self.llm.chat_stream(messages, max_tokens=max_tok):
                    response_parts.append(chunk)
                response = "".join(response_parts)
            else:
                result = self.llm.chat(messages, max_tokens=max_tok)
                response = result.get("message", {}).get("content", "")
        except Exception as e:
            response = f"[Hermes Error] LLM call failed: {e}"
        log.debug("llm: %.2fs", time.time() - t_llm)

        # Persist assistant response
        self.memory.add_message(tab_id, "assistant", response)
        self.session.append_message(tab_id, "assistant", response)

        # 3. Store in cache for next time
        if use_cache and response and not response.startswith("[Hermes Error]"):
            self.cache.put(user_message, response)

        log.debug("TOTAL: %.2fs", time.time() - t0)
        return response

    def chat_stream(self, user_message: str, use_cache: bool = True,
                    tab_id: Optional[str] = None):
        """
        Generator that yields response tokens as they're generated.
        Yields (kind, content) tuples where kind in:
          - "quick"  : final quick-reply response (one chunk)
          - "cached" : final cached response (one chunk)
          - "token"  : a token chunk from LLM
          - "done"   : final chunk, signals completion
        """
        if not self.llm.health():
            yield ("token", "[Hermes Error] Local LLM is not ready.")
            yield ("done", None)
            return

        if tab_id is None:
            tab_id = self.current_conversation
        if tab_id is None or self.session.get_tab(tab_id) is None:
            new_tab = self.session.create_tab()
            tab_id = new_tab["id"]
            self._current_conv_id = tab_id

        # Quick reply
        qr = try_quick_reply(user_message)
        if qr is not None:
            self.memory.add_message(tab_id, "user", user_message)
            self.memory.add_message(tab_id, "assistant", qr)
            self.session.append_message(tab_id, "user", user_message)
            self.session.append_message(tab_id, "assistant", qr)
            yield ("quick", qr)
            yield ("done", None)
            return

        # Cache
        if use_cache:
            cached = self.cache.get(user_message)
            if cached is not None:
                self.memory.add_message(tab_id, "user", user_message)
                self.memory.add_message(tab_id, "assistant", cached)
                self.session.append_message(tab_id, "user", user_message)
                self.session.append_message(tab_id, "assistant", cached)
                yield ("cached", cached)
                yield ("done", None)
                return

        # LLM
        self.memory.add_message(tab_id, "user", user_message)
        self.session.append_message(tab_id, "user", user_message)
        system_context = self._build_context(user_message)
        history = self.memory.get_recent_messages(tab_id, n=8)
        messages = [{"role": "system", "content": system_context}]
        for m in history[:-1]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_message})

        max_tok = self.dynamic_max_tokens(user_message)
        response_parts = []
        try:
            for chunk in self.llm.chat_stream(messages, max_tokens=max_tok):
                response_parts.append(chunk)
                yield ("token", chunk)
            response = "".join(response_parts)
        except Exception as e:
            response = f"[Hermes Error] LLM call failed: {e}"
            yield ("token", response)

        self.memory.add_message(tab_id, "assistant", response)
        self.session.append_message(tab_id, "assistant", response)

        if use_cache and response and not response.startswith("[Hermes Error]"):
            self.cache.put(user_message, response)

        yield ("done", None)

    def teach_lesson(self, lesson: str) -> str:
        """Add a lesson to memory (will be injected into future prompts)."""
        return self.memory.add_lesson(lesson, source="user")

    def teach_skill(self, skill_data: Dict) -> str:
        """Add a skill to the skills system."""
        skill = self.skills.add(skill_data)
        return skill.name

    def teach_document(self, text: str, source: str = "user") -> List[str]:
        """Ingest a document into the knowledge base."""
        ids = self.kb.add_text(text, source=source)
        self.kb.save()
        return ids

    def status(self) -> Dict:
        info = self.llm.model_info()
        from core.file_uploads import get_uploader
        from core.quick_replies import stats as qr_stats
        return {
            "llm": {
                "model": info["model_file"],
                "healthy": self.llm.health(),
                "info": info,
            },
            "knowledge_base": self.kb.stats(),
            "skills": {
                "count": len(self.skills),
                "list": self.skills.list_all(),
            },
            "memory": self.memory.stats(),
            "cache": self.cache.stats(),
            "quick_replies": qr_stats(),
            "uploads": get_uploader().stats(),
            "session": self.session.stats(),
            "skill_search": self.skill_search.stats(),
            "library": self.library.stats(),
            "checkpoint": self.checkpoint_mgr.stats(),
            "memory_store": self.memory_store.stats(),
            "hybrid_search": self.hybrid_search.stats(),
            "multi_agent": self.multi_agent.stats(),
        }


_orch: Optional[HermesOrchestrator] = None


def get_orchestrator() -> HermesOrchestrator:
    global _orch
    if _orch is None:
        _orch = HermesOrchestrator()
    return _orch


if __name__ == "__main__":
    o = get_orchestrator()
    print(json.dumps(o.status(), indent=2, ensure_ascii=False))
