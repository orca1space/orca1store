"""
Hermes Typed State
Pydantic-style state validation without external dependencies.
Inspired by LangGraph's TypedDict + Pydantic v3.
Pure local. No external services.

Features:
- BaseState: typed state container with field validation
- Field types: str, int, float, bool, list, dict, optional
- Validation on set
- Type coercion
- Schema export
"""
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class FieldType:
    """A field definition with type and constraints."""
    def __init__(self, type_: Type, required: bool = True,
                 default: Any = None, description: str = ""):
        self.type = type_
        self.required = required
        self.default = default
        self.description = description

    def validate(self, value: Any) -> Any:
        """Validate and coerce the value. Raises ValueError on failure."""
        if value is None:
            if self.required and self.default is None:
                raise ValueError("Required field is None")
            return self.default
        # Coerce
        try:
            if self.type is bool and isinstance(value, int):
                value = bool(value)
            elif self.type is int and isinstance(value, float) and value.is_integer():
                value = int(value)
            elif self.type is float and isinstance(value, int):
                value = float(value)
            else:
                value = self.type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot coerce {value!r} to {self.type.__name__}: {e}")
        return value


class BaseState:
    """
    Base class for typed state with field validation.
    Use as a base class and define fields as class attributes.
    """
    # Subclasses define: field_name = FieldType(...)
    pass

    @classmethod
    def get_schema(cls) -> Dict[str, Dict]:
        """Get the schema of all fields."""
        schema = {}
        for name in dir(cls):
            value = getattr(cls, name)
            if isinstance(value, FieldType):
                schema[name] = {
                    "type": value.type.__name__,
                    "required": value.required,
                    "default": value.default,
                    "description": value.description,
                }
        return schema

    def __init__(self, **kwargs):
        cls = type(self)
        # Init all fields with defaults
        for name in dir(cls):
            value = getattr(cls, name)
            if isinstance(value, FieldType):
                if name in kwargs:
                    setattr(self, name, value.validate(kwargs[name]))
                elif value.default is not None:
                    setattr(self, name, value.default)
                else:
                    setattr(self, name, None)
        # Apply any extra kwargs (allow override)
        for k, v in kwargs.items():
            if hasattr(cls, k) and isinstance(getattr(cls, k), FieldType):
                setattr(self, k, getattr(cls, k).validate(v))
            else:
                setattr(self, k, v)

    def validate_all(self) -> List[str]:
        """Validate all fields. Returns list of errors (empty = valid)."""
        errors = []
        cls = type(self)
        for name in dir(cls):
            value = getattr(cls, name)
            if isinstance(value, FieldType):
                actual = getattr(self, name, None)
                try:
                    value.validate(actual)
                except ValueError as e:
                    errors.append(f"{name}: {e}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def snapshot(self) -> Dict[str, Any]:
        return {"_class": self.__class__.__name__, **self.to_dict()}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"


# === Field shortcut ===
def field(type_: Type, required: bool = True, default: Any = None,
          description: str = ""):
    """Shorthand to create a field."""
    return FieldType(type_, required, default, description)


def optional(type_: Type, default: Any = None, description: str = ""):
    """Optional field shorthand."""
    return FieldType(type_, required=False, default=default, description=description)


# === Example typed states ===

class AgentState(BaseState):
    """State for a single agent execution."""
    task: FieldType = field(str, description="The current task")
    context: FieldType = field(dict, default={}, description="Context dict")
    result: FieldType = optional(str, description="Final result")
    iteration: FieldType = field(int, default=0, description="Iteration count")
    error: FieldType = optional(str, description="Error if any")
    history: FieldType = field(list, default=[], description="Message history")


class GraphExecutionState(BaseState):
    """State for a graph execution."""
    execution_id: FieldType = field(str, description="Unique execution ID")
    current_node: FieldType = optional(str, description="Current node")
    values: FieldType = field(dict, default={}, description="State values")
    checkpoint_id: FieldType = optional(str, description="Last checkpoint")
    interrupted: FieldType = field(bool, default=False, description="Was interrupted?")
    finished: FieldType = field(bool, default=False, description="Is finished?")

    def transition(self, node_name: str, new_values: dict = None) -> "GraphExecutionState":
        """Record a transition to a new node, appending to history."""
        history = list(self.values.get("_transitions", []))
        history.append({
            "from": self.current_node,
            "to": node_name,
            "ts": __import__("time").time(),
        })
        merged = dict(self.values or {})
        if new_values:
            merged.update(new_values)
        merged["_transitions"] = history
        return GraphExecutionState(
            execution_id=self.execution_id,
            current_node=node_name,
            values=merged,
            checkpoint_id=self.checkpoint_id,
            interrupted=self.interrupted,
            finished=self.finished,
        )

    @property
    def history(self) -> list:
        return list((self.values or {}).get("_transitions", []))


if __name__ == "__main__":
    s = AgentState(task="Hello", context={"x": 1})
    print("Agent state:", s)
    print("Schema:", AgentState.get_schema())
    print("Validation errors:", s.validate_all())

    g = GraphExecutionState(execution_id="exec_1", values={"key": "value"})
    print("Graph state:", g)
