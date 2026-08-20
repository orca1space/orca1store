"""
HERMES-AI | المرحلة 4: التحقق الشامل متعدد الطبقات
=====================================================
Enhanced Multi-Layer Verification (v2 - production grade)

Fixes applied over the reference module:
1. The reference module mixed sync helpers with async layer functions and depended
   on a `Decision` class that was never defined. Here the decision is a plain dict
   with an explicit schema and every layer is async-safe.
2. The original anomaly layer used a hard-coded mean of 0.8; here the baseline is
   learned from the verification history with a fallback.
3. Added an explicit verification schema so callers can validate their decision
   records before running the pipeline.
4. Each layer gets its own timeout so a single stuck check never blocks the rest.

Layers (10): syntax, logic, ethics(policy), resources, historical precedent,
impact simulation, anomaly detection, consistency, confidence threshold,
compliance.

Pure Python + asyncio. Local-only.
"""
import asyncio
import json
import re
import statistics
import time
from typing import Any, Dict, List, Optional

DEFAULT_LAYER_TIMEOUT = 0.5  # seconds per layer


class MultiLayerVerification:
    """Ten-layer verification with learned anomaly baselines."""

    LAYER_NAMES = [
        "syntax_validation",
        "logic_validation",
        "ethics_check",
        "resource_availability",
        "historical_precedent",
        "impact_simulation",
        "anomaly_detection",
        "consistency_validation",
        "confidence_threshold",
        "regulatory_compliance",
    ]

    REQUIRED_FIELDS = {"decision_id", "action"}
    VALID_ACTIONS = {
        "buy", "sell", "hold", "read", "write", "execute", "skip",
        "approve", "reject", "research", "compute", "call_llm",
    }

    def __init__(self, layer_timeout: float = DEFAULT_LAYER_TIMEOUT) -> None:
        self.layer_timeout = layer_timeout
        self.confidence_baseline = 0.8
        self.history: List[Dict[str, Any]] = []
        self.layer_pass_counts: Dict[str, int] = {n: 0 for n in self.LAYER_NAMES}
        self.layer_run_counts: Dict[str, int] = {n: 0 for n in self.LAYER_NAMES}
        self._layer_funcs = {
            "syntax_validation": self._validate_syntax,
            "logic_validation": self._validate_logic,
            "ethics_check": self._validate_ethics,
            "resource_availability": self._validate_resources,
            "historical_precedent": self._validate_historical,
            "impact_simulation": self._validate_impact,
            "anomaly_detection": self._detect_anomalies,
            "consistency_validation": self._validate_consistency,
            "confidence_threshold": self._validate_confidence,
            "regulatory_compliance": self._validate_compliance,
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def validate_schema(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-flight schema validation (sync, cheap)."""
        missing = [f for f in self.REQUIRED_FIELDS if f not in decision]
        action = decision.get("action")
        invalid_action = action is not None and action not in self.VALID_ACTIONS
        return {
            "passed": not missing and not invalid_action,
            "missing_fields": missing,
            "invalid_action": invalid_action,
            "fields_present": list(decision.keys()),
        }

    async def comprehensive_verification(
        self, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all 10 layers and return the aggregate result."""
        schema = self.validate_schema(decision)
        if not schema["passed"]:
            return {
                "all_passed": False,
                "details": {"schema": schema},
                "pass_rate": 0.0,
                "blocked_by": "schema",
            }

        results: Dict[str, Any] = {}
        for layer_name in self.LAYER_NAMES:
            fn = self._layer_funcs[layer_name]
            try:
                result = await asyncio.wait_for(
                    fn(decision), timeout=self.layer_timeout
                )
            except asyncio.TimeoutError:
                result = {"passed": False, "reason": "layer_timeout"}
            except Exception as exc:  # noqa: BLE001
                result = {"passed": False, "reason": str(exc)}
            if not isinstance(result, dict):
                result = {"passed": False, "reason": "layer_returned_invalid"}
            result.setdefault("passed", False)
            results[layer_name] = result
            self.layer_run_counts[layer_name] += 1
            if result["passed"]:
                self.layer_pass_counts[layer_name] += 1

        passed_count = sum(1 for r in results.values() if r["passed"])
        total = len(results)
        pass_rate = passed_count / total

        self.history.append(
            {
                "decision_id": decision.get("decision_id"),
                "timestamp": time.time(),
                "pass_rate": pass_rate,
                "action": decision.get("action"),
                "confidence": decision.get("confidence"),
            }
        )

        # update learned baseline
        if len(self.history) >= 10:
            rates = [h["pass_rate"] for h in self.history[-100:]]
            self.confidence_baseline = statistics.mean(rates)

        return {
            "all_passed": passed_count == total,
            "details": results,
            "pass_rate": pass_rate,
            "passed_layers": passed_count,
            "total_layers": total,
        }

    # ------------------------------------------------------------------
    # original 6 layers
    # ------------------------------------------------------------------
    async def _validate_syntax(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        did = decision.get("decision_id", "")
        if not isinstance(did, str) or not re.match(r"^[\w\-:.]{3,64}$", did):
            return {"passed": False, "reason": "invalid decision_id format"}
        action = decision.get("action")
        if not isinstance(action, str) or len(action) > 64:
            return {"passed": False, "reason": "invalid action field"}
        return {"passed": True}

    async def _validate_logic(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        reasoning = decision.get("reasoning", "")
        if not reasoning or len(str(reasoning)) < 4:
            return {"passed": False, "reason": "reasoning too short or missing"}
        # basic sanity: reasoning should not be pure numeric gibberish
        if re.fullmatch(r"[\d\s\.\,]+", str(reasoning)):
            return {"passed": False, "reason": "reasoning is numeric only"}
        return {"passed": True}

    async def _validate_ethics(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Local policy layer: checks user-defined blocked patterns."""
        policy = decision.get("_policy", {})
        blocked = policy.get("blocked_patterns") or []
        text = " ".join(
            str(decision.get(k, ""))
            for k in ("action", "reasoning", "target")
            if decision.get(k)
        ).lower()
        hits = [p for p in blocked if p.lower() in text]
        if hits:
            return {"passed": False, "reason": f"policy blocked: {hits}"}
        return {"passed": True}

    async def _validate_resources(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        limits = decision.get("_resource_limits") or {}
        if not limits:
            return {"passed": True, "reason": "no limits defined"}
        conf = decision.get("confidence", 0)
        if isinstance(conf, (int, float)) and conf < (limits.get("min_confidence", 0)):
            return {"passed": False, "reason": "confidence below resource minimum"}
        return {"passed": True}

    async def _validate_historical(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Precedent check: does a similar past decision exist with good outcomes?"""
        action = decision.get("action")
        similar = [
            h for h in self.history[-50:]
            if h.get("action") == action and h.get("pass_rate", 0) >= 0.7
        ]
        if not self.history:
            return {"passed": True, "reason": "first decision, no history yet"}
        return {
            "passed": len(similar) > 0,
            "similar_successful": len(similar),
        }

    async def _validate_impact(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate impact: high-confidence decisions with large stated cost are risky."""
        confidence = decision.get("confidence", 0)
        cost = decision.get("cost") or decision.get("impact_level", 0)
        if isinstance(confidence, (int, float)) and isinstance(cost, (int, float)):
            risk = cost * (1.0 - max(0.0, min(1.0, confidence)))
            if risk > 0.6:
                return {"passed": False, "reason": f"high impact risk ({risk:.2f})"}
        return {"passed": True}

    # ------------------------------------------------------------------
    # 4 new layers
    # ------------------------------------------------------------------
    async def _detect_anomalies(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Confidence anomaly detection against the learned baseline.

        Callers may supply an explicit baseline via
        ``_confidence_baseline`` (used by the engine so cold-start decisions
        are not punished by the static default).
        """
        confidence = decision.get("confidence")
        if not isinstance(confidence, (int, float)):
            return {"passed": True, "reason": "no numeric confidence to audit"}
        baseline = decision.get("_confidence_baseline", self.confidence_baseline)
        # when no real history exists yet, accept any plausible confidence
        if len(self.history) < 3:
            return {"passed": True, "reason": "no baseline history yet (cold start)"}
        threshold = 0.35 * max(0.2, min(0.5, baseline))
        if abs(confidence - baseline) > threshold:
            return {
                "passed": False,
                "reason": f"confidence {confidence:.2f} deviates from baseline "
                          f"{self.confidence_baseline:.2f}",
            }
        return {"passed": True}

    async def _validate_consistency(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        reasoning = str(decision.get("reasoning", "")).lower()
        action = str(decision.get("action", "")).lower()
        if not reasoning or not action:
            return {"passed": True, "reason": "nothing to compare"}
        # 1) direct token presence
        if action in reasoning:
            return {"passed": True, "similarity": 1.0, "match": "direct"}
        # 2) semantic keyword families tying actions to decision language
        families = {
            "buy": ("price", "dropped", "below", "support", "upward", "volume",
                    "cheap", "entry", "long", "ascend", "bull"),
            "sell": ("price", "rose", "above", "resistance", "downward", "top",
                     "expensive", "exit", "short", "descend", "bear"),
            "hold": ("waiting", "clearer", "signal", "stable", "uncertain",
                     "flat", "neutral", "sideways"),
            "execute": ("run", "operation", "deploy", "apply", "perform"),
            "read": ("open", "file", "content", "text", "scan", "parse"),
            "write": ("create", "save", "generate", "produce", "document"),
            "skip": ("skip", "ignore", "defer", "later"),
            "approve": ("approve", "allow", "permit", "grant"),
            "reject": ("reject", "deny", "refuse", "block"),
            "research": ("research", "search", "look", "find", "gather", "study"),
            "compute": ("compute", "calculate", "calculate", "evaluate", "solve"),
            "call_llm": ("llm", "model", "generate", "answer", "respond"),
        }
        tokens = set(reasoning.split())
        family = families.get(action, ())
        overlap = len(set(family) & tokens)
        passed = overlap >= 2
        return {
            "passed": passed,
            "similarity": min(1.0, overlap / max(1, len(family))),
            "match": "semantic_family" if passed else "none",
        }

    async def _validate_confidence(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        conf = decision.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            return {"passed": False, "reason": "confidence missing or out of [0,1]"}
        return {"passed": True}

    async def _validate_compliance(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Local regulatory/compliance layer driven by user-provided rules."""
        rules = decision.get("_compliance_rules") or []
        for rule in rules:
            if callable(rule):
                try:
                    if not rule(decision):
                        return {"passed": False, "reason": "custom rule failed"}
                except Exception as exc:  # noqa: BLE001
                    return {"passed": False, "reason": str(exc)}
        return {"passed": True}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _calculate_similarity(self, dict1: Dict, dict2: Dict) -> float:
        common = set(dict1.keys()) & set(dict2.keys())
        if not common:
            return 0.0
        total = 0.0
        for key in common:
            v1, v2 = dict1[key], dict2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                denom = max(abs(v1), abs(v2))
                total += (1.0 - abs(v1 - v2) / denom) if denom else 1.0
            else:
                s1, s2 = str(v1).lower(), str(v2).lower()
                if s1 == s2:
                    total += 1.0
                elif not s1 or not s2:
                    total += 0.0
                else:
                    # token overlap (Jaccard)
                    a, b = set(s1.split()), set(s2.split())
                    union = len(a | b)
                    total += (len(a & b) / union) if union else 0.0
        return total / len(common)

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "subsystem": "multilayer_verification",
            "layers": len(self.LAYER_NAMES),
            "verifications_run": len(self.history),
            "confidence_baseline": round(self.confidence_baseline, 3),
            "layer_pass_rates": {
                n: (
                    self.layer_pass_counts[n] / max(1, self.layer_run_counts[n])
                )
                for n in self.LAYER_NAMES
            },
        }

    def clear(self) -> None:
        self.history.clear()
        self.layer_pass_counts = {n: 0 for n in self.LAYER_NAMES}
        self.layer_run_counts = {n: 0 for n in self.LAYER_NAMES}
        self.confidence_baseline = 0.8


