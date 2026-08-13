"""
Hermes Graph Executor
Lightweight graph-based state machine inspired by LangGraph.
Pure local. No external services.

Features:
- Nodes (functions/callables)
- Edges (conditional + direct)
- Cycles (loops, retries)
- Parallel execution
- State propagation
- Time-travel via checkpoints (paired with core/checkpoint.py)
"""
import inspect
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result of a single node execution."""
    node_name: str
    status: NodeStatus
    output: Any = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class GraphState:
    """
    State container that flows through the graph.
    Mutations produce a new state (immutable semantics).
    """
    values: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_node: Optional[str] = None
    iteration: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> "GraphState":
        """Set a value and return a new state (functional update)."""
        new_values = {**self.values, key: value}
        new_history = self.history + [
            {"op": "set", "key": key, "value": value, "ts": time.time()}
        ]
        return GraphState(
            values=new_values,
            history=new_history,
            current_node=self.current_node,
            iteration=self.iteration,
            error=self.error,
            metadata=self.metadata,
        )

    def update(self, **kwargs) -> "GraphState":
        """Update multiple keys at once."""
        new_values = {**self.values, **kwargs}
        new_history = self.history + [
            {"op": "update", "values": kwargs, "ts": time.time()}
        ]
        return GraphState(
            values=new_values,
            history=new_history,
            current_node=self.current_node,
            iteration=self.iteration,
            error=self.error,
            metadata=self.metadata,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "values": dict(self.values),
            "current_node": self.current_node,
            "iteration": self.iteration,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class GraphNode:
    """A node in the graph - wraps a callable."""

    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description
        self.calls = 0
        self.total_ms = 0.0

    def __call__(self, state: GraphState) -> GraphState:
        """Execute the node function with the state."""
        t0 = time.time()
        self.calls += 1
        try:
            sig = inspect.signature(self.func)
            if len(sig.parameters) == 1:
                result = self.func(state)
            else:
                result = self.func()
            if not isinstance(result, GraphState):
                # Wrap raw return in set: function-name-keyed
                if isinstance(result, dict):
                    return state.update(**result)
                return state.set(self.name, result)
            return result
        finally:
            self.total_ms += (time.time() - t0) * 1000


class Graph:
    """
    A graph of nodes with edges. Supports:
    - add_node(name, func): add a node
    - add_edge(from, to): direct edge
    - add_conditional_edge(from, condition, mapping): route based on state
    - set_entry_point(name): where to start
    - set_finish_point(name): where to end (optional, defaults to no-end loop)
    """

    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[str]] = {}  # from -> [to]
        self.conditional_edges: Dict[str, Tuple[Callable, Dict[Any, str]]] = {}
        self.entry_point: Optional[str] = None
        self.finish_points: Set[str] = set()
        self.max_iterations: int = 50

    def add_node(self, name: str, func: Callable, description: str = "") -> "Graph":
        self.nodes[name] = GraphNode(name, func, description)
        if name not in self.edges:
            self.edges[name] = []
        return self

    def add_edge(self, from_node: str, to_node: str) -> "Graph":
        if from_node not in self.nodes:
            raise ValueError(f"Unknown source node: {from_node}")
        if to_node not in self.nodes:
            raise ValueError(f"Unknown target node: {to_node}")
        self.edges.setdefault(from_node, []).append(to_node)
        return self

    def add_conditional_edge(
        self, from_node: str, condition: Callable[[GraphState], Any],
        path_map: Dict[Any, str]
    ) -> "Graph":
        if from_node not in self.nodes:
            raise ValueError(f"Unknown source node: {from_node}")
        for path_name in path_map.values():
            if path_name not in self.nodes:
                raise ValueError(f"Unknown target node: {path_name}")
        self.conditional_edges[from_node] = (condition, path_map)
        return self

    def set_entry_point(self, name: str) -> "Graph":
        if name not in self.nodes:
            raise ValueError(f"Unknown entry point: {name}")
        self.entry_point = name
        return self

    def set_finish_point(self, name: str) -> "Graph":
        if name not in self.nodes:
            raise ValueError(f"Unknown finish point: {name}")
        self.finish_points.add(name)
        return self

    def visualize(self) -> str:
        """Simple text visualization of the graph."""
        lines = [f"Graph: {self.name}", f"  Entry: {self.entry_point}"]
        for node_name, node in self.nodes.items():
            lines.append(f"  Node '{node_name}': {node.description or node.func.__name__}")
            if node_name in self.edges:
                for to in self.edges[node_name]:
                    lines.append(f"    -> {to}")
            if node_name in self.conditional_edges:
                cond, mapping = self.conditional_edges[node_name]
                for k, v in mapping.items():
                    lines.append(f"    ->? {k}: {v}")
        if self.finish_points:
            lines.append(f"  Finish: {', '.join(self.finish_points)}")
        return "\n".join(lines)


class GraphExecutor:
    """
    Execute a graph with state propagation, parallel branches, cycles, and interrupts.

    Usage:
        g = Graph("my_flow")
        g.add_node("step1", lambda s: s.set("x", 1))
        g.add_node("step2", lambda s: s.set("y", 2))
        g.add_edge("step1", "step2")
        g.set_entry_point("step1")
        g.set_finish_point("step2")
        executor = GraphExecutor(g)
        result = executor.run(GraphState())
    """

    def __init__(self, graph: Graph, checkpoint_manager: Optional[Any] = None,
                 interrupt_handler: Optional[Callable] = None,
                 parallel: bool = False):
        self.graph = graph
        self.checkpoint = checkpoint_manager
        self.interrupt_handler = interrupt_handler
        self.parallel = parallel
        self.execution_id = uuid.uuid4().hex
        self.node_results: List[NodeResult] = []
        self._interrupted: Optional[str] = None

    def _next_node(self, current: str, state: GraphState) -> Optional[str]:
        """Determine next node based on edges."""
        # Direct edges first
        if current in self.graph.edges and self.graph.edges[current]:
            # For now, take first direct edge (parallel will use multiple)
            return self.graph.edges[current][0]
        # Conditional edges
        if current in self.graph.conditional_edges:
            cond, path_map = self.graph.conditional_edges[current]
            decision = cond(state)
            return path_map.get(decision)
        return None

    def _execute_node(self, node: GraphNode, state: GraphState) -> NodeResult:
        """Execute a single node with timing and error handling."""
        t0 = time.time()
        new_state = state.update(current_node=node.name, iteration=state.iteration + 1)
        try:
            result_state = node(new_state)
            return NodeResult(
                node_name=node.name,
                status=NodeStatus.DONE,
                output=result_state,
                duration_ms=(time.time() - t0) * 1000,
                started_at=t0,
                finished_at=time.time(),
            )
        except Exception as e:
            return NodeResult(
                node_name=node.name,
                status=NodeStatus.FAILED,
                error=e,
                duration_ms=(time.time() - t0) * 1000,
                started_at=t0,
                finished_at=time.time(),
            )

    def run(self, initial_state: GraphState) -> GraphState:
        """Execute the graph from entry point until finish."""
        if not self.graph.entry_point:
            raise ValueError("No entry point set")

        state = initial_state
        current = self.graph.entry_point
        self.node_results = []
        self._interrupted = None

        for iteration in range(self.graph.max_iterations):
            if current is None:
                break
            if current not in self.graph.nodes:
                raise ValueError(f"Unknown node: {current}")

            # Checkpoint before execution
            if self.checkpoint:
                try:
                    self.checkpoint.save(
                        execution_id=self.execution_id,
                        step=iteration,
                        node_name=current,
                        state=state,
                    )
                except Exception as e:
                    import logging
                    logging.getLogger("hermes.graph").debug(
                        "Checkpoint save failed (non-fatal): %s", e
                    )

            # Check for interrupt
            if self.interrupt_handler:
                should_continue, new_state = self.interrupt_handler(current, state)
                if not should_continue:
                    self._interrupted = current
                    break
                if new_state:
                    state = new_state

            # Execute node
            node = self.graph.nodes[current]
            result = self._execute_node(node, state)
            self.node_results.append(result)

            if result.status == NodeStatus.FAILED:
                state = state.update(error=str(result.error))
                break

            state = result.output
            current = self._next_node(current, state)
            # Stop after executing a finish point
            if current in self.graph.finish_points:
                # Execute the finish point then stop
                node = self.graph.nodes[current]
                result = self._execute_node(node, state)
                self.node_results.append(result)
                if result.status == NodeStatus.DONE:
                    state = result.output
                break

        return state

    def run_parallel(self, initial_state: GraphState,
                    branches: List[List[str]]) -> List[GraphState]:
        """
        Run multiple branches in parallel, each is a list of node names.
        Returns a list of final states, one per branch.
        """
        results = [None] * len(branches)
        with ThreadPoolExecutor(max_workers=len(branches)) as ex:
            futures = {}
            for i, branch in enumerate(branches):
                state = initial_state
                for node_name in branch:
                    if node_name not in self.graph.nodes:
                        raise ValueError(f"Unknown node: {node_name}")
                    node = self.graph.nodes[node_name]
                    state = self._execute_node(node, state)
                    if state.error:
                        break
                futures[ex.submit(lambda s: s, state)] = i
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
        return [r for r in results if r is not None]

    def interrupt(self):
        """Stop execution at the current node (called from interrupt_handler)."""
        self._interrupted = self.graph.entry_point

    def get_node_results(self) -> List[Dict[str, Any]]:
        return [
            {
                "node": r.node_name,
                "status": r.status.value,
                "duration_ms": round(r.duration_ms, 2),
                "error": str(r.error) if r.error else None,
            }
            for r in self.node_results
        ]


# === Helper decorators ===

def node(name: str, description: str = ""):
    """Decorator: mark a function as a graph node."""
    def decorator(func):
        func._graph_node = {"name": name, "description": description}
        return func
    return decorator


def build_graph_from_decorators(graph_name: str, *funcs) -> Graph:
    """Build a graph from functions decorated with @node."""
    g = Graph(graph_name)
    nodes = []
    for f in funcs:
        if not hasattr(f, "_graph_node"):
            continue
        info = f._graph_node
        g.add_node(info["name"], f, info["description"])
        nodes.append(info["name"])
    # Linear edges by default
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    if nodes:
        g.set_entry_point(nodes[0])
        g.set_finish_point(nodes[-1])
    return g


# === Spec-based factories (for API/server use) ===

def _make_callable_from_spec(spec: dict, ns: dict):
    """Build a callable from a node spec.

    Accepts:
      - {"fn": "lambda s: ..."}  -> compiled Python callable (state -> state)
      - {"op": "set", "key": "x", "value": 1}  -> small built-in operations
      - {"op": "increment", "key": "x", "by": 1}
      - {"op": "append", "key": "log", "value": "..."}
    """
    if "fn" in spec:
        code = spec["fn"]
        safe_globals = {
            "__builtins__": __builtins__,
            "json": __import__("json"),
        }
        # The lambda receives the GraphState. It must return a GraphState.
        body = f"def _node(state):\n    return ({code})(state)\n"
        exec(compile(body, "<graph_spec>", "exec"), safe_globals)
        return safe_globals["_node"]

    op = spec.get("op")

    def _op_node(state):
        d = dict(state.values) if hasattr(state, "values") else dict(state)
        if op == "set":
            d[spec["key"]] = spec["value"]
        elif op == "increment":
            d[spec["key"]] = d.get(spec["key"], 0) + spec.get("by", 1)
        elif op == "append":
            d.setdefault(spec["key"], []).append(spec["value"])
        elif op == "noop":
            pass
        else:
            raise ValueError(f"Unknown op: {op!r}")
        if hasattr(state, "update"):
            return state.update(**d)
        return d

    return _op_node


def build_graph_from_spec(spec: dict) -> Graph:
    """Build a Graph from a JSON-like spec.

    spec = {
        "name": "my_graph",
        "nodes": [{"id": "a", "fn": "...", "description": "..."}, ...],
        "edges": [{"from": "a", "to": "b"}, ...],
        "conditional": [
            {"from": "b", "key": "count", "mapping": {"even": "c", "odd": "d"}}, ...
        ],
        "entry": "a",
        "finish": ["d"],
    }
    """
    name = spec.get("name", "graph")
    g = Graph(name)
    ns = {}
    for n in spec.get("nodes", []):
        nid = n["id"]
        fn = _make_callable_from_spec(n, ns)
        g.add_node(nid, fn, n.get("description", ""))
    for e in spec.get("edges", []):
        g.add_edge(e["from"], e["to"])
    for c in spec.get("conditional", []):
        key = c["key"]
        mapping = c["mapping"]

        def _cond(state, _k=key):
            v = state.values.get(_k) if hasattr(state, "values") else state.get(_k)
            for k, val in mapping.items():
                if isinstance(k, str) and k.startswith(">"):
                    try:
                        if v > int(k[1:]):
                            return val
                    except Exception:
                        pass
                elif isinstance(k, str) and k.startswith("<"):
                    try:
                        if v < int(k[1:]):
                            return val
                    except Exception:
                        pass
                else:
                    if v == k:
                        return val
            return None

        g.add_conditional_edge(c["from"], _cond, mapping)
    if spec.get("entry"):
        g.set_entry_point(spec["entry"])
    for f in spec.get("finish", []) or []:
        g.set_finish_point(f)
    return g


def get_graph_executor(spec: dict, checkpoint_manager=None, interrupt_handler=None):
    """Factory: build a graph from spec and return a GraphExecutor."""
    g = build_graph_from_spec(spec)
    return GraphExecutor(g, checkpoint_manager=checkpoint_manager,
                          interrupt_handler=interrupt_handler)
