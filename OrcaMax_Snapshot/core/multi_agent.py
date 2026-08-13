"""
Hermes Multi-Agent System
Multiple specialized agents working together.
Inspired by LangGraph's supervisor pattern + AutoGen's group chat.
Pure local. No external services.

Features:
- Agent class (role + system prompt + tools)
- Supervisor pattern (routes between agents)
- Handoff (agents transfer control)
- Parallel execution (multiple agents at once)
- Shared state across agents
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import HERMES_ROOT


AGENTS_DIR = HERMES_ROOT / "data" / "agents"


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    PLANNER = "planner"
    CUSTOM = "custom"


@dataclass
class AgentMessage:
    """A message in the agent conversation."""
    role: str  # "user" | "agent_name" | "supervisor"
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content,
                "timestamp": self.timestamp, "metadata": self.metadata}


@dataclass
class Agent:
    """A specialized agent."""
    name: str
    role: AgentRole
    system_prompt: str
    description: str = ""
    tools: List[str] = field(default_factory=list)  # skill names
    parent: Optional[str] = None  # for hierarchical agents
    can_handoff_to: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _calls: int = 0
    _total_ms: float = 0.0

    def call(self, state: Dict) -> Dict:
        """Stub: actual call happens via orchestrator. Returns its output dict."""
        self._calls += 1
        # Real implementation in orchestrator
        return {"agent": self.name, "result": "ok"}


class MultiAgentOrchestrator:
    """
    Manages a team of agents.
    - Supervisor pattern: a supervisor agent routes to specialized agents
    - Handoff: agents can transfer control to others
    - Parallel: run multiple agents simultaneously
    - Shared state: all agents see the same state
    """

    def __init__(self, name: str = "team"):
        self.name = name
        self.agents: Dict[str, Agent] = {}
        self.conversations: Dict[str, List[AgentMessage]] = {}
        self._orchestrator_fn: Optional[Callable] = None

    def add_agent(self, agent: Agent) -> "MultiAgentOrchestrator":
        self.agents[agent.name] = agent
        if agent.name not in self.conversations:
            self.conversations[agent.name] = []
        return self

    def set_supervisor(self, agent: Agent) -> "MultiAgentOrchestrator":
        """Mark an agent as the supervisor (router)."""
        agent.role = AgentRole.SUPERVISOR
        return self.add_agent(agent)

    def set_orchestrator_fn(self, fn: Callable) -> "MultiAgentOrchestrator":
        """Set the actual LLM-based orchestrator function."""
        self._orchestrator_fn = fn
        return self

    def handoff(self, from_agent: str, to_agent: str) -> "MultiAgentOrchestrator":
        """Allow from_agent to transfer control to to_agent."""
        if from_agent in self.agents:
            if to_agent not in self.agents[from_agent].can_handoff_to:
                self.agents[from_agent].can_handoff_to.append(to_agent)
        return self

    def post_message(self, agent_name: str, message: AgentMessage):
        """Add a message to an agent's conversation history."""
        if agent_name not in self.conversations:
            self.conversations[agent_name] = []
        self.conversations[agent_name].append(message)

    def get_conversation(self, agent_name: str) -> List[AgentMessage]:
        return self.conversations.get(agent_name, [])

    def get_state(self) -> Dict[str, Any]:
        """Get shared state across all agents."""
        return {
            "agents": {n: {
                "name": a.name,
                "role": a.role.value,
                "description": a.description,
                "tools": a.tools,
                "calls": a._calls,
                "can_handoff_to": a.can_handoff_to,
            } for n, a in self.agents.items()},
            "messages": {n: [m.to_dict() for m in msgs]
                          for n, msgs in self.conversations.items()},
        }

    def list_agents(self) -> List[Dict]:
        return [{
            "name": a.name,
            "role": a.role.value,
            "description": a.description,
            "tools": a.tools,
            "calls": a._calls,
        } for a in self.agents.values()]

    def run_sequential(self, agent_names: List[str], initial_message: str,
                       orchestrator_fn: Optional[Callable] = None) -> List[AgentMessage]:
        """Run a sequence of agents, each processing the output of the previous."""
        if orchestrator_fn is None:
            orchestrator_fn = self._orchestrator_fn
        if orchestrator_fn is None:
            raise ValueError("No orchestrator function set")

        history = [AgentMessage("user", initial_message, time.time())]
        for agent_name in agent_names:
            if agent_name not in self.agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            agent = self.agents[agent_name]
            t0 = time.time()
            response = orchestrator_fn(agent, history)
            agent._total_ms += (time.time() - t0) * 1000
            agent._calls += 1
            msg = AgentMessage(agent.name, response, time.time(),
                              {"duration_ms": (time.time() - t0) * 1000})
            history.append(msg)
            self.post_message(agent_name, msg)
        return history

    def run_supervisor(self, initial_message: str,
                       orchestrator_fn: Optional[Callable] = None,
                       max_steps: int = 10) -> List[AgentMessage]:
        """
        Supervisor pattern: supervisor agent routes to specialized agents.
        The supervisor decides which agent to call next.
        """
        if orchestrator_fn is None:
            orchestrator_fn = self._orchestrator_fn
        if orchestrator_fn is None:
            raise ValueError("No orchestrator function set")

        supervisor = None
        for a in self.agents.values():
            if a.role == AgentRole.SUPERVISOR:
                supervisor = a
                break
        if supervisor is None:
            raise ValueError("No supervisor agent set")

        history = [AgentMessage("user", initial_message, time.time())]
        current_agent_name = supervisor.name
        for step in range(max_steps):
            if current_agent_name not in self.agents:
                break
            agent = self.agents[current_agent_name]
            t0 = time.time()
            decision = orchestrator_fn(agent, history)
            agent._total_ms += (time.time() - t0) * 1000
            agent._calls += 1

            # Decision can be: {"next": "agent_name", "content": "..."} or {"finish": "..."}
            if isinstance(decision, dict):
                content = decision.get("content", "")
                next_agent = decision.get("next")
                if decision.get("finish"):
                    history.append(AgentMessage(current_agent_name, content, time.time()))
                    self.post_message(current_agent_name, history[-1])
                    break
            else:
                content = str(decision)
                next_agent = None

            history.append(AgentMessage(current_agent_name, content, time.time()))
            self.post_message(current_agent_name, history[-1])

            if not next_agent or next_agent not in self.agents:
                break
            current_agent_name = next_agent
        return history

    def run_parallel(self, agent_names: List[str], message: str,
                    orchestrator_fn: Optional[Callable] = None) -> List[List[AgentMessage]]:
        """Run multiple agents in parallel, each sees the same input."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if orchestrator_fn is None:
            orchestrator_fn = self._orchestrator_fn
        if orchestrator_fn is None:
            raise ValueError("No orchestrator function set")

        results = [None] * len(agent_names)
        with ThreadPoolExecutor(max_workers=len(agent_names)) as ex:
            futures = {}
            for i, name in enumerate(agent_names):
                agent = self.agents[name]
                history = [AgentMessage("user", message, time.time())]
                futures[ex.submit(orchestrator_fn, agent, history)] = (i, name)
            for fut in as_completed(futures):
                i, name = futures[fut]
                content = fut.result()
                if isinstance(content, dict):
                    content = content.get("content", str(content))
                msg = AgentMessage(name, content, time.time())
                self.post_message(name, msg)
                self.agents[name]._calls += 1
                results[i] = [msg]
        return [r for r in results if r is not None]

    def stats(self) -> Dict:
        return {
            "name": self.name,
            "agents": len(self.agents),
            "total_messages": sum(len(m) for m in self.conversations.values()),
            "total_calls": sum(a._calls for a in self.agents.values()),
        }


def make_default_agents() -> Dict[str, Agent]:
    """Create a standard team of agents."""
    return {
        "supervisor": Agent(
            name="supervisor",
            role=AgentRole.SUPERVISOR,
            system_prompt="You are a supervisor agent. Route tasks to the right specialist.",
            description="Routes tasks between agents",
            can_handoff_to=["researcher", "coder", "reviewer"],
        ),
        "researcher": Agent(
            name="researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="You are a research agent. Find information and answer questions.",
            description="Researches and finds information",
            tools=["src_pdf", "src_docx", "src_webapp_testing"],
        ),
        "coder": Agent(
            name="coder",
            role=AgentRole.CODER,
            system_prompt="You are a coding agent. Write, debug, and review code.",
            description="Writes and reviews code",
            tools=["src_algorithmic_art", "src_webapp_testing"],
        ),
        "reviewer": Agent(
            name="reviewer",
            role=AgentRole.REVIEWER,
            system_prompt="You are a reviewer agent. Check work for quality and correctness.",
            description="Reviews work for quality",
        ),
    }


# Singleton
_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_multi_agent() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator("hermes_team")
        for agent in make_default_agents().values():
            _orchestrator.add_agent(agent)
    return _orchestrator


if __name__ == "__main__":
    mo = get_multi_agent()
    print("Multi-agent:", mo.stats())
    for a in mo.list_agents():
        print(f"  - {a['name']} ({a['role']}): {a['description']}")
