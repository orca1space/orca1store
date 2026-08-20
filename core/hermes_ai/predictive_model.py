"""
HERMES-AI | المرحلة 5: النموذج التنبؤي المتقدم (Ensemble)
=====================================================
Enhanced Predictive Model (v2 - production grade)

Fixes applied over the reference module:
1. The reference sub-models were literal placeholders returning magic constants.
   Here each sub-model computes a real estimate from the observed series
   (mean, trend extrapolation, AR-style weighted history, pattern projection),
   so the ensemble output actually responds to data.
2. Weights are normalized and dynamically re-balance toward sub-models with
   better recent accuracy (simple online calibration).
3. Adds an accuracy tracker so calibration converges as real outcomes arrive.

Pure Python + asyncio. Local-only.
"""
import asyncio
import statistics
import time
from collections import deque
from typing import Any, Dict, List, Optional

TARGET_FIELD = "target"
DEFAULT_WEIGHTS = {
    "mean": 0.25,
    "trend": 0.25,
    "ar": 0.25,
    "pattern": 0.25,
}
MIN_OBSERVATIONS = 3


class PredictiveModel:
    """Ensemble prediction from four calibrated sub-models."""

    def __init__(self, history_capacity: int = 2000) -> None:
        self.series: deque = deque(maxlen=history_capacity)
        self.weights = dict(DEFAULT_WEIGHTS)
        self.accuracies: Dict[str, List[float]] = {k: [] for k in DEFAULT_WEIGHTS}
        self.predictions_made = 0
        self.calibrations = 0

    # ------------------------------------------------------------------
    # observation / calibration
    # ------------------------------------------------------------------
    def observe(self, value: float) -> None:
        """Feed a realized value; used to calibrate sub-model accuracy."""
        if isinstance(value, (int, float)):
            self.series.append(float(value))

    def record_outcome(self, model_name: str, predicted: float, actual: float) -> None:
        """Register how accurate a sub-model prediction was (0..1 score)."""
        denom = max(1e-9, max(abs(predicted), abs(actual)))
        accuracy = max(0.0, 1.0 - abs(predicted - actual) / denom)
        self.accuracies.setdefault(model_name, []).append(accuracy)
        if len(self.accuracies[model_name]) >= 15:
            self._rebalance_weights()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def predict_with_ensemble(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the four sub-models and merge with dynamic weights."""
        merged = {**context, **data}  # data overrides context for field lookups
        value = data.get(TARGET_FIELD) or data.get("price") or data.get("value")
        if not isinstance(value, (int, float)):
            value = None
        predictions: Dict[str, Optional[Dict[str, Any]]] = {
            "mean": await self._sub_mean(value, merged),
            "trend": await self._sub_trend(value, merged),
            "ar": await self._sub_ar(value, merged),
            "pattern": await self._sub_pattern(value, merged),
        }

        combined = self._combine_predictions(predictions)
        combined["method"] = "ensemble"
        combined["sources"] = sum(1 for p in predictions.values() if p)
        combined["predicted_at"] = time.time()
        combined["_data_value_used"] = value
        self.predictions_made += 1
        return combined

    # ------------------------------------------------------------------
    # sub-models (real computations over the observed series)
    # ------------------------------------------------------------------
    async def _sub_mean(
        self, value: Optional[float], data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.series:
            return None
        values = list(self.series)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        confidence = min(0.9, 0.4 + 0.05 * min(len(values), 10))
        return {"prediction": mean, "confidence": confidence, "stdev": stdev}

    async def _sub_trend(
        self, value: Optional[float], data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if len(self.series) < 3:
            return None
        values = list(self.series)
        last3 = values[-3:]
        slope = (last3[-1] - last3[0]) / 2.0
        prediction = last3[-1] + slope
        confidence = min(0.85, 0.35 + 0.05 * min(len(values), 10))
        return {"prediction": prediction, "confidence": confidence}

    async def _sub_ar(
        self, value: Optional[float], data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """AR-style: weighted average of the last 5 observations (newest heaviest)."""
        if len(self.series) < 2:
            return None
        values = list(self.series)[-5:]
        weights = [(i + 1) for i in range(len(values))]
        total_w = sum(weights)
        prediction = sum(v * w for v, w in zip(values, weights)) / total_w
        confidence = min(0.82, 0.3 + 0.06 * min(len(values), 5))
        return {"prediction": prediction, "confidence": confidence}

    async def _sub_pattern(
        self, value: Optional[float], data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Project the stored prototype pattern scaled to current magnitude."""
        patterns = data.get("_patterns") or data.get("patterns") or []
        if not patterns or not self.series:
            return None
        current_mean = statistics.mean(list(self.series))
        best: Optional[Dict[str, Any]] = None
        for pattern in patterns:
            if not isinstance(pattern, (list, tuple)) or len(pattern) < 2:
                continue
            if not all(isinstance(x, (int, float)) for x in pattern):
                continue
            pattern_mean = statistics.mean(pattern)
            if pattern_mean == 0:
                continue
            scale = current_mean / pattern_mean
            candidate = pattern[-1] * scale
            # score by similarity of the scaled penultimate value
            sim = 1.0 - min(1.0, abs(list(self.series)[-1] - pattern[-1] * scale)
                            / max(1e-9, abs(list(self.series)[-1])))
            if best is None or sim > best["_sim"]:
                best = {"prediction": candidate, "confidence": 0.4 + 0.4 * sim, "_sim": sim}
        return best

    # ------------------------------------------------------------------
    # ensemble merge
    # ------------------------------------------------------------------
    def _combine_predictions(
        self, predictions: Dict[str, Optional[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        active = {k: v for k, v in predictions.items() if v}
        if not active:
            # no data at all: neutral prediction
            return {
                "prediction": 0.5,
                "confidence": 0.25,
                "sources": 0,
                "weights_used": {},
                "note": "insufficient_data",
            }

        # normalize weights to active sub-models
        total_w = sum(self.weights.get(k, 0.25) for k in active)
        weighted_sum = 0.0
        total_conf = 0.0
        weights_used: Dict[str, float] = {}
        for key, pred in active.items():
            w = (self.weights.get(key, 0.25) / total_w) if total_w else 1.0 / len(active)
            weighted_sum += pred["prediction"] * w
            total_conf += pred["confidence"] * w
            weights_used[key] = round(w, 3)

        return {
            "prediction": weighted_sum,
            "confidence": total_conf,
            "weights_used": weights_used,
            "sub_predictions": {k: v["prediction"] for k, v in active.items()},
        }

    def _rebalance_weights(self) -> None:
        """Shift weight toward sub-models with better recent accuracy."""
        scores: Dict[str, float] = {}
        for name in DEFAULT_WEIGHTS:
            recent = self.accuracies.get(name, [])[-20:]
            scores[name] = statistics.mean(recent) if recent else 0.5
        total = sum(scores.values())
        if total <= 0:
            return
        self.weights = {k: v / total for k, v in scores.items()}
        self.calibrations += 1

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "subsystem": "predictive_model",
            "observations": len(self.series),
            "predictions_made": self.predictions_made,
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "calibrations": self.calibrations,
            "recent_accuracies": {
                k: round(statistics.mean(v[-20:]), 3) if v else None
                for k, v in self.accuracies.items()
            },
        }

    def clear(self) -> None:
        self.series.clear()
        self.weights = dict(DEFAULT_WEIGHTS)
        self.accuracies = {k: [] for k in DEFAULT_WEIGHTS}
        self.predictions_made = 0
        self.calibrations = 0
