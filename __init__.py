"""
HERMES-AI — الحزمة الموحدة للنظام المحسّن
=====================================================
Unified package that wires the five optimization phases into one engine:

  Phase 1  concurrent_inference  — race-condition-free concurrent inference
  Phase 2  data_completion       — intelligent missing-data completion
  Phase 3  recovery_center       — 12-strategy escalating self-healing
  Phase 4  multilayer_verification — 10-layer decision verification
  Phase 5  predictive_model      — calibrated ensemble predictions

The `HermesAIEngine` orchestrates all five and exposes a single decision
pipeline: complete -> infer -> predict -> verify -> (recover on failure).
"""
from core.hermes_ai.concurrent_inference import ConcurrentInferenceSubsystem
from core.hermes_ai.data_completion import IntelligentDataCompletion
from core.hermes_ai.recovery_center import AdvancedRecoveryCenter
from core.hermes_ai.multilayer_verification import MultiLayerVerification
from core.hermes_ai.predictive_model import PredictiveModel
from core.hermes_ai.engine import HermesAIEngine

__all__ = [
    "ConcurrentInferenceSubsystem",
    "IntelligentDataCompletion",
    "AdvancedRecoveryCenter",
    "MultiLayerVerification",
    "PredictiveModel",
    "HermesAIEngine",
]

__version__ = "2.0.0"
