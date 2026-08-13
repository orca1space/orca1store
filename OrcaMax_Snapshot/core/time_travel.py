"""
Hermes Time Travel
Replay and fork from any historical checkpoint.
Inspired by LangGraph's time travel feature.
Pure local. No external services.

Features:
- list_history(thread): all checkpoints for a thread
- get_state_at(thread, step): state at specific step
- fork(thread, step): create a new branch from a checkpoint
- rollback(thread, step): restore to a previous state
- diff(thread, step1, step2): show state changes between steps
"""
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.checkpoint import CheckpointManager, get_checkpoint_manager, Checkpoint


class TimeTravel:
    """Time travel / replay operations on checkpoint history."""

    def __init__(self, manager: Optional[CheckpointManager] = None):
        self.manager = manager or get_checkpoint_manager()

    def list_history(self, execution_id: str) -> List[Dict[str, Any]]:
        """Return the full checkpoint history for an execution."""
        cps = self.manager.get_history(execution_id)
        return [
            {
                "id": cp.id,
                "step": cp.step,
                "node": cp.node_name,
                "created_at": cp.created_at,
                "parent": cp.parent_id,
                "has_state": bool(cp.state),
                "pending_writes": len(cp.pending_writes),
            }
            for cp in cps
        ]

    def get_state_at(self, execution_id: str, step: int) -> Optional[Dict]:
        """Return the state snapshot at a given step."""
        cps = self.manager.get_history(execution_id)
        for cp in cps:
            if cp.step == step:
                return {
                    "step": cp.step,
                    "node": cp.node_name,
                    "state": cp.state,
                    "created_at": cp.created_at,
                }
        return None

    def fork(self, source_execution_id: str, fork_step: int,
             new_execution_id: Optional[str] = None) -> Optional[str]:
        """
        Create a new execution branch from a checkpoint.
        Returns the new execution_id, or None if the step not found.
        """
        cps = self.manager.get_history(source_execution_id)
        for cp in cps:
            if cp.step == fork_step:
                new_id = new_execution_id or uuid.uuid4().hex
                # Save the checkpoint under the new execution id
                cp.id = uuid.uuid4().hex
                cp.execution_id = new_id
                cp.parent_id = f"{source_execution_id}:{fork_step}"
                cp.metadata["forked_from"] = {
                    "execution": source_execution_id,
                    "step": fork_step,
                    "original_id": cp.id,
                }
                self.manager.backend.save(cp)
                return new_id
        return None

    def rollback(self, execution_id: str, step: int) -> Optional[Dict]:
        """
        Get the state to roll back to.
        Returns the state, or None if not found.
        (The actual rollback is the caller's responsibility - they apply this state.)
        """
        return self.get_state_at(execution_id, step)

    def diff(self, execution_id: str, step1: int, step2: int) -> Dict:
        """
        Show state changes between two steps.
        Returns keys added, removed, changed.
        """
        s1 = self.get_state_at(execution_id, step1)
        s2 = self.get_state_at(execution_id, step2)
        if not s1 or not s2:
            return {"error": "Step not found"}
        v1 = s1["state"].get("values", s1["state"])
        v2 = s2["state"].get("values", s2["state"])
        added = {k: v2[k] for k in v2 if k not in v1}
        removed = {k: v1[k] for k in v1 if k not in v2}
        changed = {
            k: {"from": v1[k], "to": v2[k]}
            for k in v1 if k in v2 and v1[k] != v2[k]
        }
        return {
            "from_step": step1,
            "to_step": step2,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def replay(self, execution_id: str, from_step: int = 0) -> List[Dict]:
        """Get all states from from_step onwards for replay."""
        cps = self.manager.get_history(execution_id)
        return [
            {"step": cp.step, "node": cp.node_name, "state": cp.state}
            for cp in cps if cp.step >= from_step
        ]

    def get_branches(self, execution_id: str) -> List[Dict]:
        """List all forked branches from this execution."""
        cps = self.manager.get_history(execution_id)
        branches = []
        for cp in cps:
            if "forked_from" in cp.metadata:
                branches.append({
                    "checkpoint_id": cp.id,
                    "from": cp.metadata["forked_from"],
                    "step": cp.step,
                })
        return branches

    def search_history(self, execution_id: str, query: str) -> List[Dict]:
        """Search through state snapshots for a query string."""
        query_lower = query.lower()
        results = []
        for entry in self.list_history(execution_id):
            cp = self.manager.get_by_id(entry["id"])
            if cp and query_lower in str(cp.state).lower():
                results.append({
                    "step": entry["step"],
                    "node": entry["node"],
                    "snippet": str(cp.state)[:200],
                })
        return results

    def stats(self, execution_id: str) -> Dict:
        """Get statistics for an execution's history."""
        history = self.list_history(execution_id)
        if not history:
            return {"count": 0, "duration": 0}
        times = [h["created_at"] for h in history]
        return {
            "count": len(history),
            "first_step": history[0]["step"],
            "last_step": history[-1]["step"],
            "duration": times[-1] - times[0] if len(times) > 1 else 0,
            "branches": len(self.get_branches(execution_id)),
        }


# Singleton
_tt: Optional[TimeTravel] = None


def get_time_travel() -> TimeTravel:
    global _tt
    if _tt is None:
        _tt = TimeTravel()
    return _tt


if __name__ == "__main__":
    tt = get_time_travel()
    cm = get_checkpoint_manager()

    # Create some test checkpoints
    eid = "demo_exec"
    cm.delete_execution(eid)
    for i in range(5):
        cm.save(eid, i, f"node_{i}", {"values": {"x": i * 10}}, metadata={"step": i})

    print("History:", tt.list_history(eid))
    print("State at step 2:", tt.get_state_at(eid, 2))
    print("Diff (0->3):", tt.diff(eid, 0, 3))
    print("Stats:", tt.stats(eid))

    # Fork
    new_eid = tt.fork(eid, 2, "demo_fork")
    print(f"Forked: {new_eid}")
    cm.delete_execution(eid)
    if new_eid:
        cm.delete_execution(new_eid)
