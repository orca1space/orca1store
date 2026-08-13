"""
Hermes Human-in-the-Loop
Interrupt graph execution and require human approval before continuing.
Inspired by LangGraph's interrupt() pattern.
Pure local. No external services.

Features:
- interrupt(reason, state): pause execution and ask for approval
- approve(resume_state): continue from where we left off
- reject(reason): abort with reason
- modify(new_state): change state and continue
"""
import time
import uuid
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT


INTERRUPTS_DIR = HERMES_ROOT / "data" / "interrupts"


class InterruptDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class InterruptRequest:
    """A request to pause execution and ask for human input."""

    def __init__(self, interrupt_id: str, execution_id: str, node_name: str,
                 reason: str, state: Dict, context: Optional[Dict] = None):
        self.id = interrupt_id
        self.execution_id = execution_id
        self.node_name = node_name
        self.reason = reason
        self.state = state
        self.context = context or {}
        self.created_at = time.time()
        self.resolved_at: Optional[float] = None
        self.decision: Optional[InterruptDecision] = None
        self.resolution: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "node_name": self.node_name,
            "reason": self.reason,
            "state": self.state,
            "context": self.context,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "decision": self.decision.value if self.decision else None,
            "resolution": self.resolution,
        }


class HumanInTheLoop:
    """Manager for interrupt-based human approval gates."""

    def __init__(self):
        self.interrupts_dir = INTERRUPTS_DIR
        self.interrupts_dir.mkdir(parents=True, exist_ok=True)
        self._pending: Dict[str, InterruptRequest] = {}
        # Lock protects _pending dict + pending.json file across HTTP threads
        self._lock = threading.RLock()

    def request(self, execution_id: str, node_name: str, reason: str,
               state: Any, context: Optional[Dict] = None) -> InterruptRequest:
        """Create an interrupt request."""
        if hasattr(state, "snapshot"):
            state_dict = state.snapshot()
        elif isinstance(state, dict):
            state_dict = state
        else:
            state_dict = {"_value": str(state)}
        req = InterruptRequest(
            interrupt_id=uuid.uuid4().hex,
            execution_id=execution_id,
            node_name=node_name,
            reason=reason,
            state=state_dict,
            context=context,
        )
        with self._lock:
            self._pending[req.id] = req
            self._save_pending()
        return req

    def approve(self, interrupt_id: str) -> bool:
        """Approve the interrupt: continue with current state."""
        with self._lock:
            self._load_pending()
            if interrupt_id not in self._pending:
                return False
            req = self._pending[interrupt_id]
            req.decision = InterruptDecision.APPROVE
            req.resolved_at = time.time()
            req.resolution = {"approved": True}
            del self._pending[interrupt_id]
            self._save_pending()
            return True

    def reject(self, interrupt_id: str, reason: str = "") -> bool:
        """Reject the interrupt: abort execution."""
        with self._lock:
            self._load_pending()
            if interrupt_id not in self._pending:
                return False
            req = self._pending[interrupt_id]
            req.decision = InterruptDecision.REJECT
            req.resolved_at = time.time()
            req.resolution = {"rejected": True, "reason": reason}
            del self._pending[interrupt_id]
            self._save_pending()
            return True

    def modify(self, interrupt_id: str, new_state: Any) -> bool:
        """Modify state and continue."""
        with self._lock:
            if interrupt_id not in self._pending:
                return False
            req = self._pending[interrupt_id]
            req.decision = InterruptDecision.MODIFY
            req.resolved_at = time.time()
            if hasattr(new_state, "snapshot"):
                new_state_dict = new_state.snapshot()
            else:
                new_state_dict = new_state
            req.resolution = {"modified": True, "new_state": new_state_dict}
            del self._pending[interrupt_id]
            self._save_pending()
            return True

    def get_pending(self, interrupt_id: str) -> Optional[InterruptRequest]:
        with self._lock:
            return self._pending.get(interrupt_id)

    def list_pending(self, refresh: bool = True) -> List[Dict]:
        with self._lock:
            if refresh:
                # Reload from disk so we see requests created by other processes
                self._load_pending()
            return [req.to_dict() for req in self._pending.values()]

    def has_pending(self, execution_id: str) -> bool:
        with self._lock:
            return any(req.execution_id == execution_id
                       for req in self._pending.values())

    def _save_pending(self):
        """Persist pending interrupts to disk (so user can decide offline)."""
        import json
        with open(self.interrupts_dir / "pending.json", "w", encoding="utf-8") as f:
            data = [r.to_dict() for r in self._pending.values()]
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_pending(self):
        """Load pending interrupts from disk."""
        import json
        path = self.interrupts_dir / "pending.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                req = InterruptRequest(
                    interrupt_id=d["id"],
                    execution_id=d["execution_id"],
                    node_name=d["node_name"],
                    reason=d["reason"],
                    state=d["state"],
                    context=d.get("context", {}),
                )
                req.created_at = d.get("created_at", time.time())
                self._pending[req.id] = req
        except (json.JSONDecodeError, KeyError, OSError):
            pass


# Default interrupt handler for GraphExecutor
def default_interrupt_handler(node_name: str, state) -> tuple:
    """
    Default handler: no interrupt, always continue.
    Returns (should_continue, new_state).
    Replace with HumanInTheLoop for real approval gates.
    """
    return True, None


_hitl: Optional[HumanInTheLoop] = None


def get_hitl() -> HumanInTheLoop:
    global _hitl
    if _hitl is None:
        _hitl = HumanInTheLoop()
        _hitl._load_pending()
    return _hitl


if __name__ == "__main__":
    hitl = get_hitl()
    req = hitl.request("exec_demo", "test_node", "Approve execution?", {"x": 1})
    print("Created:", req.id[:8])
    print("Pending:", len(hitl.list_pending()))
    hitl.approve(req.id)
    print("After approve:", len(hitl.list_pending()))
