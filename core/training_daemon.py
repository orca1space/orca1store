"""
Hermes Training Daemon
======================

A self-contained background training system that:
- Connects to external APIs / libraries / repos for 10 hours (configurable)
- Acquires new skills, knowledge, lessons, memory entries
- Persists EVERYTHING to disk continuously (survives crashes / restarts)
- Falls back to local self-training (using the local LLM) when the budget runs out
- Runs independently of any external orchestrator — once started, it
  continues without supervision.

DESIGN PRINCIPLES:
  1. **Persistent state**: every task result is saved to disk before moving on.
  2. **Resume-friendly**: if the daemon or the OS restarts, training picks up
     where it left off.
  3. **Budget-aware**: a wall-clock budget (default 10h) controls how long
     the daemon will keep training. After the budget is exhausted, the
     daemon continues with LOCAL self-training (no external calls).
  4. **Independent**: the daemon runs inside the Hermes process (or as a
     standalone process). It does NOT depend on any other service.
  5. **Auditable**: every action is logged to disk.
"""
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT
from core.logging_setup import get_logger

log = get_logger("training_daemon")

# === Persistent state file ===
TRAINING_DIR = HERMES_ROOT / "data" / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = TRAINING_DIR / "training_state.json"
LOG_FILE = TRAINING_DIR / "training.log"

# === Default sources (10 hours of material) ===
# Each entry: {"type": ..., "config": ...}
# The daemon will work through this queue.
DEFAULT_SOURCES = [
    {"type": "kb_topic", "config": {"topic": "Python best practices", "questions": 30}},
    {"type": "kb_topic", "config": {"topic": "Arabic language nuances", "questions": 20}},
    {"type": "kb_topic", "config": {"topic": "Windows command line", "questions": 25}},
    {"type": "kb_topic", "config": {"topic": "SQL fundamentals", "questions": 25}},
    {"type": "self_train", "config": {"rounds": 50, "per_round": 5}},
    {"type": "skill_usage_train", "config": {"per_skill": 5}},
    {"type": "skill_index_rebuild", "config": {}},
    {"type": "system_prompt_train", "config": {"max_additions": 3}},
]

@dataclass
class Task:
    """A single training task."""
    id: str
    type: str          # "hf_dataset" | "kb_topic" | "self_train" | "api_import" | "github_repo"
    config: Dict
    status: str = "pending"  # pending | running | done | failed
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    items_imported: int = 0
    notes: str = ""


@dataclass
class DaemonState:
    started_at: float
    budget_seconds: float = 36000.0   # 10 hours default
    external_budget_used: float = 0.0
    local_train_count: int = 0
    tasks: List[Task] = field(default_factory=list)
    paused: bool = False
    finished: bool = False

    def time_left(self) -> float:
        if self.finished:
            return 0.0
        elapsed = time.time() - self.started_at
        return max(0.0, self.budget_seconds - elapsed)


class TrainingDaemon:
    """
    Background training daemon. Persists state to disk.
    Runs as a thread inside the webui process.
    """

    def __init__(self, state_file: Path = STATE_FILE,
                 log_file: Path = LOG_FILE):
        self.state_file = state_file
        self.log_file = log_file
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state: Optional[DaemonState] = None
        # Also log to file
        self._file_handler = None
        self._setup_file_logging()

    def _setup_file_logging(self):
        """Log all daemon activity to data/training/training.log."""
        try:
            self._file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            self._file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            log.addHandler(self._file_handler)
        except OSError:
            pass

    # ===== State persistence =====

    def _load_state(self) -> DaemonState:
        if not self.state_file.exists():
            return None
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            tasks = [Task(**t) for t in raw.get("tasks", [])]
            return DaemonState(
                started_at=raw["started_at"],
                budget_seconds=raw.get("budget_seconds", 36000.0),
                external_budget_used=raw.get("external_budget_used", 0.0),
                local_train_count=raw.get("local_train_count", 0),
                tasks=tasks,
                paused=raw.get("paused", False),
                finished=raw.get("finished", False),
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("Could not load state: %s. Starting fresh.", e)
            return None

    def _save_state(self):
        with self._lock:
            if self._state is None:
                return
            raw = {
                "started_at": self._state.started_at,
                "budget_seconds": self._state.budget_seconds,
                "external_budget_used": self._state.external_budget_used,
                "local_train_count": self._state.local_train_count,
                "tasks": [asdict(t) for t in self._state.tasks],
                "paused": self._state.paused,
                "finished": self._state.finished,
                "last_saved": time.time(),
            }
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.state_file)

    # ===== Public API =====

    def start(self, budget_seconds: float = 36000.0,
              sources: Optional[List[Dict]] = None,
              resume: bool = True) -> Dict:
        """
        Start the training daemon. Returns a summary.

        - If `resume=True` and there's an existing state, continue it.
        - If `resume=False`, start fresh.
        - If already running, just return current status.
        """
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"already_running": True, "status": self.status()}

            if resume and (self._state is None):
                self._state = self._load_state()

            if self._state is None:
                # Fresh start
                self._state = DaemonState(
                    started_at=time.time(),
                    budget_seconds=budget_seconds,
                )
                for src in (sources or DEFAULT_SOURCES):
                    task = Task(
                        id=f"task_{len(self._state.tasks):04d}",
                        type=src["type"],
                        config=src.get("config", {}),
                    )
                    self._state.tasks.append(task)
                self._save_state()
                log.info("Daemon started fresh with %d tasks, budget=%.0fs",
                         len(self._state.tasks), budget_seconds)
            else:
                # Resume only when work or budget remains.
                pending = any(t.status in ("pending", "running") for t in self._state.tasks)
                if self._state.finished or (not pending and self._state.time_left() <= 0):
                    self._state.finished = True
                    self._state.paused = False
                    self._save_state()
                    log.info("Daemon already complete; no budget remains.")
                    return {"started": False, "finished": True, "status": self.status()}
                self._state.paused = False
                self._state.finished = False
                self._save_state()
                log.info("Daemon resumed: %d tasks, %d done, %.0fs left",
                         len(self._state.tasks),
                         sum(1 for t in self._state.tasks if t.status == "done"),
                         self._state.time_left())

        # Start the thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="training-daemon")
        self._thread.start()
        return {"started": True, "status": self.status()}

    def pause(self) -> Dict:
        with self._lock:
            if self._state:
                self._state.paused = True
                self._save_state()
        return {"paused": True, "status": self.status()}

    def resume(self) -> Dict:
        return self.start(resume=True)

    def stop(self) -> Dict:
        """Stop the daemon (can be restarted later)."""
        self._stop_event.set()
        return {"stopped": True, "status": self.status()}

    def add_sources(self, sources: List[Dict]) -> Dict:
        """Append new sources to the queue."""
        with self._lock:
            if self._state is None:
                return {"error": "daemon not started"}
            for src in sources:
                task = Task(
                    id=f"task_{len(self._state.tasks):04d}",
                    type=src["type"],
                    config=src.get("config", {}),
                )
                self._state.tasks.append(task)
            self._save_state()
        return {"added": len(sources), "total_tasks": len(self._state.tasks)}

    def status(self) -> Dict:
        with self._lock:
            if self._state is None:
                return {"running": False, "state": "not started"}
            elapsed = time.time() - self._state.started_at
            done = sum(1 for t in self._state.tasks if t.status == "done")
            failed = sum(1 for t in self._state.tasks if t.status == "failed")
            pending = sum(1 for t in self._state.tasks if t.status == "pending")
            running = sum(1 for t in self._state.tasks if t.status == "running")
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "paused": self._state.paused,
                "finished": self._state.finished,
                "started_at": self._state.started_at,
                "elapsed_seconds": elapsed,
                "budget_seconds": self._state.budget_seconds,
                "time_left_seconds": self._state.time_left(),
                "local_train_count": self._state.local_train_count,
                "tasks_total": len(self._state.tasks),
                "tasks_pending": pending,
                "tasks_running": running,
                "tasks_done": done,
                "tasks_failed": failed,
                "last_task": self._state.tasks[-1].id if self._state.tasks else None,
            }

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ===== Main loop =====

    def _run(self):
        """Main daemon loop. Runs in background thread."""
        log.info("Daemon thread started")
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    if self._state is None or self._state.finished:
                        log.info("Daemon state is None or finished, exiting loop")
                        break
                    if self._state.paused:
                        # Wait, then re-check
                        self._stop_event.wait(2.0)
                        continue
                    # Find next pending task
                    task = next(
                        (t for t in self._state.tasks if t.status == "pending"),
                        None
                    )
                    time_left = self._state.time_left()

                if task is None:
                    # All tasks done. Do self-training with remaining time.
                    if time_left > 0:
                        log.info("All planned tasks done. Starting self-training loop. Time left: %.0fs", time_left)
                        self._do_self_train_round()
                        # After self-training, stop. External sources exhausted.
                        if time_left < 60:
                            with self._lock:
                                self._state.finished = True
                                self._save_state()
                            log.info("Budget nearly exhausted. Daemon finished.")
                            break
                        continue
                    else:
                        log.info("All tasks done and no time left. Finished.")
                        with self._lock:
                            self._state.finished = True
                            self._save_state()
                        break

                # Check if we should still do external work
                if time_left < 30:
                    # Time almost up — just do local self-training
                    log.info("Time nearly up (%.0fs left). Skipping to self-training.", time_left)
                    self._do_self_train_round()
                    continue

                # Run the task
                self._execute_task(task)

            log.info("Daemon loop exited")
        except Exception as e:
            log.error("Daemon loop crashed: %s\n%s", e, traceback.format_exc())

    def _execute_task(self, task: Task):
        """Run a single task and update its state."""
        with self._lock:
            task.status = "running"
            task.started_at = time.time()
            self._save_state()

        t0 = time.time()
        try:
            if task.type in ("hf_dataset", "api_import", "github_repo"):
                log.info("Offline mode skipped external task %s [%s].", task.id, task.type)
                with self._lock:
                    task.status = "done"
                    task.completed_at = time.time()
                    task.items_imported = 0
                    self._save_state()
                return
            if task.type == "hf_dataset":
                items = self._task_hf_dataset(task.config)
            elif task.type == "kb_topic":
                items = self._task_kb_topic(task.config)
            elif task.type == "self_train":
                items = self._task_self_train(task.config)
            elif task.type == "api_import":
                items = self._task_api_import(task.config)
            elif task.type == "github_repo":
                items = self._task_github_repo(task.config)
            elif task.type == "skill_usage_train":
                items = self._task_skill_usage_train(task.config)
            elif task.type == "skill_index_rebuild":
                items = self._task_skill_index_rebuild(task.config)
            elif task.type == "system_prompt_train":
                items = self._task_system_prompt_train(task.config)
            else:
                raise ValueError(f"Unknown task type: {task.type}")

            elapsed = time.time() - t0
            with self._lock:
                task.status = "done"
                task.completed_at = time.time()
                task.items_imported = items
                self._state.external_budget_used += elapsed
                self._save_state()
            log.info("Task %s [%s] DONE: %d items in %.1fs",
                     task.id, task.type, items, elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            with self._lock:
                task.status = "failed"
                task.completed_at = time.time()
                task.error = str(e)[:300]
                self._state.external_budget_used += elapsed
                self._save_state()
            log.error("Task %s [%s] FAILED after %.1fs: %s",
                      task.id, task.type, elapsed, e)

    # ===== Task implementations =====

    def _task_hf_dataset(self, config: Dict) -> int:
        raise RuntimeError("local_only: external HF dataset disabled")
        """Download Q&A pairs from a HuggingFace dataset."""
        repo = config.get("repo", "")
        limit = int(config.get("limit", 100))
        if not repo:
            raise ValueError("missing repo")
        log.info("Downloading HF dataset: %s (limit=%d)", repo, limit)

        # Use the datasets library if available, else fall back to API
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, split="train", streaming=True,
                              trust_remote_code=True)
            count = 0
            kb_items = []
            for i, row in enumerate(ds):
                if i >= limit:
                    break
                # Try common field names
                q = (row.get("question") or row.get("instruction") or
                     row.get("input") or row.get("prompt") or "")
                a = (row.get("answer") or row.get("response") or
                     row.get("output") or row.get("completion") or "")
                if q and a and isinstance(q, str) and isinstance(a, str):
                    kb_items.append({
                        "text": f"Q: {q}\nA: {a}",
                        "source": f"hf:{repo}:{i}",
                    })
                    count += 1
            # Add to KB
            if kb_items:
                self._add_to_kb(kb_items)
            return count
        except ImportError:
            log.warning("'datasets' not installed. Falling back to API.")
            return self._task_hf_dataset_fallback(repo, limit)

    def _task_hf_dataset_fallback(self, repo: str, limit: int) -> int:
        """Use HF datasets-server API as fallback."""
        url = f"https://datasets-server.huggingface.co/rows?dataset={repo}&split=train&offset=0&length={min(limit, 100)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Hermes/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            rows = data.get("rows", [])
            kb_items = []
            for row in rows[:limit]:
                # HF API returns {row: {field: value, ...}, row_idx: N}
                fields = row.get("row", {})
                q = (fields.get("question") or fields.get("instruction") or
                     fields.get("input") or fields.get("prompt") or "")
                a = (fields.get("answer") or fields.get("response") or
                     fields.get("output") or "")
                if q and a:
                    kb_items.append({
                        "text": f"Q: {q}\nA: {a}",
                        "source": f"hf:{repo}:{row.get('row_idx', '?')}",
                    })
            if kb_items:
                self._add_to_kb(kb_items)
            return len(kb_items)
        except Exception as e:
            log.warning("HF API fallback failed: %s", e)
            return 0

    def _task_kb_topic(self, config: Dict) -> int:
        """Generate Q&A knowledge on a topic using the LOCAL LLM.

        This is the key: even if external APIs fail, the daemon can
        still teach itself using the local Qwen 3B model.
        """
        topic = config.get("topic", "")
        n_questions = int(config.get("questions", 20))
        if not topic:
            raise ValueError("missing topic")
        log.info("Self-generating %d Q&A pairs on: %s", n_questions, topic)

        from core.llm import get_llm
        llm = get_llm()
        kb_items = []
        topics_to_use = [
            f"Explain a key concept about {topic} in 2-3 sentences.",
            f"Give a practical example related to {topic}.",
            f"What is a common mistake with {topic}?",
            f"Write a short tip about {topic}.",
            f"What is a beginner question about {topic}? Answer it briefly.",
        ]
        for i in range(n_questions):
            prompt = topics_to_use[i % len(topics_to_use)]
            try:
                result = llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=200,
                )
                content = result.get("message", {}).get("content", "").strip()
                if content and len(content) > 20:
                    kb_items.append({
                        "text": f"# {topic}\n\n{content}",
                        "source": f"self_train:topic:{topic}:{i}",
                    })
            except Exception as e:
                log.warning("LLM Q&A generation failed (i=%d): %s", i, e)
            # Be a good citizen
            if self._stop_event.is_set():
                break
        if kb_items:
            self._add_to_kb(kb_items)
        return len(kb_items)

    def _task_self_train(self, config: Dict) -> int:
        """Train from existing KB: pick random chunks, ask LLM to generate
        related Q&A pairs, store as new memory entries."""
        rounds = int(config.get("rounds", 10))
        per_round = int(config.get("per_round", 3))
        log.info("Self-training from existing KB: %d rounds x %d", rounds, per_round)

        from core.llm import get_llm
        from core.knowledge import get_kb
        llm = get_llm()
        kb = get_kb()
        # Sample random chunks
        chunks = []
        if hasattr(kb.store, "entries") and kb.store.entries:
            chunks = random.sample(
                kb.store.entries,
                min(rounds * per_round, len(kb.store.entries))
            )
        else:
            return 0

        from core.memory import get_memory
        mem = get_memory()
        added = 0
        for i in range(0, len(chunks), per_round):
            batch = chunks[i:i + per_round]
            for chunk in batch:
                text = (chunk.get("content", "") or "")[:500]
                if not text:
                    continue
                prompt = (
                    f"Based on this text:\n---\n{text}\n---\n\n"
                    f"Generate one insightful question and its answer "
                    f"that would help someone learn this topic."
                )
                try:
                    result = llm.chat(
                        [{"role": "user", "content": prompt}],
                        max_tokens=200,
                    )
                    qa = result.get("message", {}).get("content", "").strip()
                    if qa and len(qa) > 30:
                        mem.add_message("self_train", "user",
                                        f"Learn: {qa[:200]}")
                        mem.add_message("self_train", "assistant",
                                        f"Understood. Added to memory.")
                        added += 1
                except Exception as e:
                    log.warning("self-train LLM call failed: %s", e)
            if self._stop_event.is_set():
                break
        with self._lock:
            self._state.local_train_count += added
            self._save_state()
        return added

    def _task_skill_usage_train(self, config: Dict) -> int:
        """OPERATIONAL TRAINING: For each skill, generate concrete
        example user queries + how the skill should respond.

        This converts skills from "passive definitions" to "active
        operational patterns" the agent can actually use.
        """
        n_examples_per_skill = int(config.get("per_skill", 5))
        skill_filter = config.get("skill_filter")  # None = all
        log.info("Skill-usage training: %d examples per skill", n_examples_per_skill)

        from core.llm import get_llm
        from core.skills import get_skills
        from core.memory import get_memory
        llm = get_llm()
        skills_mgr = get_skills()
        mem = get_memory()
        all_skills = list(skills_mgr.skills.values())
        if skill_filter:
            all_skills = [s for s in all_skills if s.name in skill_filter]

        added = 0
        for skill in all_skills:
            if self._stop_event.is_set():
                break
            prompt = (
                f"You are learning how to use a skill called '{skill.name}'.\n"
                f"Description: {skill.description}\n"
                f"Procedure: {(skill.procedure or '')[:400]}\n\n"
                f"Generate {n_examples_per_skill} realistic example user queries "
                f"where this skill should be used, and a short example response "
                f"showing how to apply the procedure.\n\n"
                f"Format each example as:\n"
                f"EXAMPLE 1\n"
                f"USER: <example user query>\n"
                f"ASSISTANT: <example response>\n"
            )
            try:
                result = llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=400,
                )
                examples = result.get("message", {}).get("content", "").strip()
                if examples and len(examples) > 50:
                    # Save to a "skill_examples" memory conversation
                    mem.add_message("skill_examples", "user",
                                    f"Examples for skill '{skill.name}':")
                    mem.add_message("skill_examples", "assistant", examples)
                    added += n_examples_per_skill
            except Exception as e:
                log.warning("skill_usage_train failed for %s: %s", skill.name, e)
        with self._lock:
            self._state.local_train_count += added
            self._save_state()
        log.info("skill_usage_train: trained %d examples across %d skills",
                 added, len(all_skills))
        return added

    def _task_system_prompt_train(self, config: Dict) -> int:
        """OPERATIONAL TRAINING: Analyze current system prompt + lessons,
        suggest concise improvements, and append to prompt_additions.txt.

        This makes Hermes progressively better at following its own rules.
        """
        max_additions = int(config.get("max_additions", 3))
        log.info("System-prompt training: up to %d new rules", max_additions)

        from core.llm import get_llm
        from core.memory import get_memory
        llm = get_llm()
        mem = get_memory()
        lessons = mem.get_lessons(limit=20)
        existing_additions_path = HERMES_ROOT / "data" / "prompt_additions.txt"
        existing = ""
        if existing_additions_path.exists():
            existing = existing_additions_path.read_text(encoding="utf-8")

        lessons_text = "\n".join(f"- {l.get('lesson', '')[:200]}" for l in lessons)
        if not lessons_text.strip():
            return 0

        prompt = (
            f"Below are recent lessons learned. Suggest {max_additions} NEW "
            f"concise rules (1 sentence each) that would help the agent apply "
            f"these lessons in the future. Output them as a JSON array of strings.\n\n"
            f"Lessons:\n{lessons_text}\n\n"
            f"Existing rules (do NOT duplicate):\n{existing}\n\n"
            f"Output format: [\"rule 1\", \"rule 2\"]\n"
        )
        try:
            result = llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            raw = result.get("message", {}).get("content", "").strip()
            # Parse JSON array
            import re
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                rules = json.loads(m.group(0))
                added = 0
                for rule in rules:
                    if isinstance(rule, str) and rule.strip() and rule.strip() not in existing:
                        with open(existing_additions_path, "a", encoding="utf-8") as f:
                            f.write(rule.strip() + "\n")
                        added += 1
                return added
        except Exception as e:
            log.warning("system_prompt_train failed: %s", e)
        return 0

    def _task_skill_index_rebuild(self, config: Dict) -> int:
        """OPERATIONAL TRAINING: For each skill, ask the LLM to generate
        rich trigger keywords + better description, then update the skill.

        This makes the semantic skill search much better at finding the
        right skill for each query.
        """
        log.info("Rebuilding skill indexes with richer triggers + descriptions")
        from core.llm import get_llm
        from core.skills import get_skills
        from core.skill_library import get_library
        llm = get_llm()
        skills_mgr = get_skills()
        library = get_library()
        updated = 0
        for skill in list(skills_mgr.skills.values()):
            if self._stop_event.is_set():
                break
            prompt = (
                f"Skill: {skill.name}\n"
                f"Current description: {skill.description}\n"
                f"Current triggers: {', '.join(skill.triggers or [])}\n\n"
                f"Generate 8 trigger keywords/phrases a user might say to invoke this skill, "
                f"and rewrite the description in 1-2 sentences. "
                f"Output as JSON: {{\"triggers\": [..], \"description\": \"...\"}}"
            )
            try:
                result = llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=200,
                )
                raw = result.get("message", {}).get("content", "").strip()
                import re
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                if m:
                    obj = json.loads(m.group(0))
                    if obj.get("triggers") and obj.get("description"):
                        # Update the skill
                        skill_dict = skill.data.copy()
                        skill_dict["trigger_keywords"] = obj["triggers"][:12]
                        skill_dict["description"] = obj["description"]
                        library.save_skill(skill_dict,
                                          source=f"trained:skill_index")
                        updated += 1
            except Exception as e:
                log.warning("skill_index_rebuild failed for %s: %s", skill.name, e)
        # Reindex embeddings
        try:
            from core.orchestrator import get_orchestrator
            o = get_orchestrator()
            if hasattr(o, "skill_search"):
                o.skill_search.index_skills(o.skills.skills, force=True)
        except Exception as e:
            log.warning("Final reindex failed: %s", e)
        return updated

    def _task_api_import(self, config: Dict) -> int:
        raise RuntimeError("local_only: external API import disabled")
        """Use the bulk API importer."""
        url = config.get("url", "")
        if not url:
            raise ValueError("missing url")
        from core.api_importer import get_api_importer
        result = get_api_importer().import_everything(
            url=url,
            api_key=config.get("api_key"),
            timeout=int(config.get("timeout", 60)),
        )
        return (
            result.get("skills", {}).get("imported", 0)
            + result.get("knowledge", {}).get("imported", 0)
            + result.get("lessons", {}).get("imported", 0)
        )

    def _task_github_repo(self, config: Dict) -> int:
        raise RuntimeError("local_only: external GitHub import disabled")
        """Shallow-clone a local Git repo and ingest docs into KB."""
        url = config.get("url", "")
        if not url:
            raise ValueError("missing url")
        from core.loaders import GitLoader
        loader = GitLoader(Path(url))
        docs = loader.load()
        if docs:
            self._add_to_kb(docs)
        return len(docs)

    def _do_self_train_round(self):
        """Run a single self-training round using existing KB."""
        try:
            self._task_self_train({"rounds": 2, "per_round": 2})
        except Exception as e:
            log.warning("self_train round failed: %s", e)

    # ===== Helpers =====

    def _add_to_kb(self, items: List[Dict]):
        """Add a list of {text, source} items to the knowledge base."""
        try:
            from core.knowledge import get_kb
            kb = get_kb()
            for item in items:
                text = item.get("text", "")
                source = item.get("source", "imported:daemon")
                if text and isinstance(text, str):
                    kb.add_text(text, source=source, metadata={"imported_by": "training_daemon"})
        except Exception as e:
            log.warning("KB add failed: %s", e)


# === Singleton ===
_daemon: Optional[TrainingDaemon] = None


def get_training_daemon() -> TrainingDaemon:
    global _daemon
    if _daemon is None:
        _daemon = TrainingDaemon()
    return _daemon
