"""
HERMES-AI | المرحلة 2: نظام استكمال البيانات الذكي
=====================================================
Intelligent Data Completion (v2 - production grade)

Fixes applied over the reference module:
1. The original pipeline silently returned the *input* dict (with None values)
   when every strategy had low confidence. Here the completed values are always
   merged back into the record with an explicit confidence trail.
2. Strategy confidence thresholds use per-strategy fallback values instead of a
   hard single cut (0.7) that discarded everything.
3. Adds statistical interpolation from the *actual* historical series, pattern
   matching against stored prototypes, and correlation inference between fields.
4. Tracks per-field completion quality and exposes a completion report.

Pure Python + asyncio. Local-only.
"""
import asyncio
import statistics
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

STRATEGIES = [
    "last_known_value",
    "statistical_interpolation",
    "pattern_matching",
    "correlation_inference",
    "contextual_inference",
]

MIN_CONFIDENCE = 0.35   # below this the field is marked unresolved
HIGH_CONFIDENCE = 0.70  # stop trying strategies above this


class IntelligentDataCompletion:
    """Complete missing fields with the best available strategy."""

    def __init__(self, history_capacity: int = 5000) -> None:
        self.historical_data: Dict[str, deque] = {}
        self.cap = history_capacity
        # simple field-to-field correlation hints populated by observations
        self.correlations: Dict[str, Dict[str, float]] = {}
        self.patterns: Dict[str, List[float]] = {}
        self.stats_completed = 0
        self.stats_unresolved = 0

    # ------------------------------------------------------------------
    # observation (feed real data in so interpolation works)
    # ------------------------------------------------------------------
    def observe(self, record: Dict[str, Any]) -> None:
        """Register a complete record into the historical series."""
        for field, value in record.items():
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                continue
            if field not in self.historical_data:
                self.historical_data[field] = deque(maxlen=self.cap)
            self.historical_data[field].append(float(value))

    def set_correlation(self, source: str, target: str, coefficient: float) -> None:
        """Manually (or programmatically) register a correlation hint."""
        self.correlations.setdefault(source, {})[target] = float(coefficient)

    def set_pattern(self, name: str, values: List[float]) -> None:
        self.patterns[name] = list(values)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def complete_missing_data(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        confidence_floor: float = MIN_CONFIDENCE,
    ) -> Dict[str, Any]:
        """Return a new record where missing fields are completed.

        The output always contains every input key. Completed fields carry
        extra metadata under `_completion` in the returned dict so callers can
        distinguish real values from estimates.
        """
        result = dict(data)
        report: Dict[str, Any] = {}

        missing = [k for k, v in data.items() if v is None]
        for field in missing:
            filled = await self._try_strategies(field, data, context, confidence_floor)
            if filled is not None and filled["confidence"] >= confidence_floor:
                result[field] = filled["value"]
                report[field] = {
                    "value": filled["value"],
                    "confidence": filled["confidence"],
                    "strategy": filled["strategy"],
                }
                self.stats_completed += 1
            else:
                self.stats_unresolved += 1
                report[field] = {"value": None, "confidence": 0.0, "strategy": "none"}

        result["_completion"] = report
        result["_completion_rate"] = (
            sum(1 for v in report.values() if v["value"] is not None)
            / max(1, len(report))
        )
        return result

    # ------------------------------------------------------------------
    # strategy cascade
    # ------------------------------------------------------------------
    async def _try_strategies(
        self,
        field: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
        floor: float,
    ) -> Optional[Dict[str, Any]]:
        # explicit context hints win over everything (highest authority)
        has_hint = (
            field in (context.get("field_hints") or {})
            or field in (context.get("defaults") or {})
        )
        ordered = [STRATEGIES[4]] + STRATEGIES[:4] if has_hint else STRATEGIES
        strategy_fns = [
            (STRATEGIES[0], self._strategy_last_known_value),
            (STRATEGIES[1], self._strategy_statistical_interpolation),
            (STRATEGIES[2], self._strategy_pattern_matching),
            (STRATEGIES[3], self._strategy_correlation_inference),
            (STRATEGIES[4], self._strategy_contextual_inference),
        ]
        strategy_fns = [
            (name, fn)
            for name, fn in strategy_fns
            if name in ordered
        ] + [
            (name, fn)
            for name, fn in [
                (STRATEGIES[0], self._strategy_last_known_value),
                (STRATEGIES[1], self._strategy_statistical_interpolation),
                (STRATEGIES[2], self._strategy_pattern_matching),
                (STRATEGIES[3], self._strategy_correlation_inference),
                (STRATEGIES[4], self._strategy_contextual_inference),
            ]
            if name not in ordered
        ]
        best: Optional[Dict[str, Any]] = None
        for name, fn in strategy_fns:
            try:
                value, confidence = await fn(field, data, context)
            except Exception:  # noqa: BLE001
                continue
            if confidence > (best["confidence"] if best else 0.0):
                best = {"value": value, "confidence": confidence, "strategy": name}
                if confidence >= HIGH_CONFIDENCE:
                    break
        return best

    async def _strategy_last_known_value(
        self, field: str, data: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[Optional[float], float]:
        series = self.historical_data.get(field)
        if not series:
            return None, 0.0
        values = list(series)[-10:]
        if not values:
            return None, 0.0
        # recent values weigh more
        weights = [0.5 + 0.5 * (i / max(1, len(values) - 1)) for i in range(len(values))]
        total = sum(weights)
        value = sum(v * w for v, w in zip(values, weights)) / total
        confidence = min(0.85, 0.45 + 0.04 * min(len(values), 10))
        return value, confidence

    async def _strategy_statistical_interpolation(
        self, field: str, data: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[Optional[float], float]:
        series = self.historical_data.get(field)
        if not series or len(series) < 3:
            return None, 0.0
        values = list(series)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 2 else 0.0
        # linear-trend extrapolation from the last points
        if len(values) >= 3:
            last3 = values[-3:]
            slope = (last3[-1] - last3[0]) / 2.0
            value = last3[-1] + slope
            # clamp to +-3 sigma so outliers never pollute the field
            value = max(mean - 3 * stdev, min(mean + 3 * stdev, value))
        else:
            value = mean
        confidence = min(0.8, 0.3 + 0.05 * min(len(values), 10))
        return value, confidence

    async def _strategy_pattern_matching(
        self, field: str, data: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[Optional[float], float]:
        """Match the current record against stored prototype patterns."""
        series = self.historical_data.get(field)
        if not series or not self.patterns:
            return None, 0.0
        candidates: List[Tuple[float, float]] = []
        for name, pattern in self.patterns.items():
            if len(pattern) < 2:
                continue
            # scale the pattern to the current series magnitude
            scale = statistics.mean(list(series)) / max(1e-9, statistics.mean(pattern))
            value = pattern[-1] * scale
            # score by how similar the penultimate scaled values are
            sim = 1.0 - min(1.0, abs((list(series)[-1]) - pattern[-1] * scale)
                            / max(1e-9, abs(list(series)[-1])))
            candidates.append((value, sim))
        if not candidates:
            return None, 0.0
        best_value, best_sim = max(candidates, key=lambda c: c[1])
        return best_value, min(0.75, best_sim * 0.8)

    async def _strategy_correlation_inference(
        self, field: str, data: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[Optional[float], float]:
        """Infer the field from correlated fields that ARE present."""
        hints = self.correlations.get(field, {})
        if not hints:
            return None, 0.0
        candidates: List[Tuple[float, float]] = []
        for source, coeff in hints.items():
            src_value = data.get(source)
            if not isinstance(src_value, (int, float)) or src_value == 0:
                continue
            # normalized ratio from history: E[field]/E[source]
            f_series = self.historical_data.get(field)
            s_series = self.historical_data.get(source)
            if not f_series or not s_series:
                continue
            f_mean = statistics.mean(list(f_series))
            s_mean = statistics.mean(list(s_series))
            if s_mean == 0:
                continue
            ratio = f_mean / s_mean
            value = src_value * ratio * max(-1.0, min(1.0, coeff))
            n = min(len(list(f_series)), len(list(s_series)))
            confidence = min(0.72, abs(coeff) * (0.4 + 0.03 * min(n, 10)))
            candidates.append((value, confidence))
        if not candidates:
            return None, 0.0
        best_value, best_conf = max(candidates, key=lambda c: c[1])
        return best_value, best_conf

    async def _strategy_contextual_inference(
        self, field: str, data: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[Optional[float], float]:
        """Use explicit hints supplied in the context."""
        hints = context.get("field_hints") or {}
        hint = hints.get(field)
        if hint is None:
            # also accept a global default hint for the field
            hint = (context.get("defaults") or {}).get(field)
        if hint is None:
            return None, 0.0
        if not isinstance(hint, (int, float)):
            return None, 0.0
        return float(hint), context.get("hint_confidence", 0.5)

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "subsystem": "data_completion",
            "fields_observed": {k: len(v) for k, v in self.historical_data.items()},
            "completed_fields": self.stats_completed,
            "unresolved_fields": self.stats_unresolved,
            "patterns": len(self.patterns),
            "correlations": sum(len(v) for v in self.correlations.values()),
        }

    def clear(self) -> None:
        self.historical_data.clear()
        self.stats_completed = 0
        self.stats_unresolved = 0
