"""
HERMES-AI — Comprehensive Test Suite
=====================================
Tests all five optimization phases plus the unified engine.
No LLM model required. Pure unit + async integration tests.

Run:
    python -m tests.hermes_ai.test_hermes_ai
    python -m tests.hermes_ai.test_hermes_ai --phase all
"""
import asyncio
import json
import sys
from pathlib import Path

HERMES_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(HERMES_ROOT))

from core.hermes_ai import (
    HermesAIEngine,
    ConcurrentInferenceSubsystem,
    IntelligentDataCompletion,
    AdvancedRecoveryCenter,
    MultiLayerVerification,
    PredictiveModel,
)

PASSED = 0
FAILED = 0
REPORT = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    status = "PASS" if ok else "FAIL"
    if ok:
        PASSED += 1
    else:
        FAILED += 1
    REPORT.append({"test": name, "status": status, "detail": detail})
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


# ══════════════════════════════════════════════════════════════════════
# Phase 1: Concurrent Inference
# ══════════════════════════════════════════════════════════════════════
async def test_phase1() -> None:
    print("\n[Phase 1] Concurrent Inference Subsystem")
    sub = ConcurrentInferenceSubsystem()

    # 1.1 basic safe inference returns merged understanding
    result = await sub.safe_concurrent_inference(
        {"price": 150, "volume": 1000}, {"market": "bullish"}
    )
    check(
        "1.1 safe inference runs and merges",
        isinstance(result.get("understanding"), dict)
        and len(result["understanding"]) >= 3,
        f"sources={result.get('_sources')}",
    )
    # 1.2 confidence is REAL now (fixed the 0% bug): merged from dict results
    check(
        "1.2 confidence_score > 0 (dict-merge bug fixed)",
        isinstance(result.get("confidence_score"), float)
        and result["confidence_score"] > 0.0,
        f"confidence={result.get('confidence_score'):.2%}",
    )
    # 1.3 deterministic id + cache hit
    r2 = await sub.safe_concurrent_inference({"price": 150, "volume": 1000}, {})
    check("1.3 deterministic cache hit", r2.get("_cache_hit") is True)

    # 1.4 race-condition stress: 40 concurrent batches, no exceptions
    async def run_batch(i: int):
        return await sub.safe_concurrent_inference(
            {"price": 100 + i, "volume": i * 10}, {}
        )

    results = await asyncio.gather(*(run_batch(i) for i in range(40)))
    check(
        "1.4 concurrent stress (40 batches)",
        all(isinstance(r, dict) for r in results),
        f"runs={sub.total_runs}",
    )
    # 1.5 fallback on total failure (timeout with 0s timeout)
    sub2 = ConcurrentInferenceSubsystem(timeout=0.0001)
    sub2._infer_market_state = sub2._infer_trend = sub2._infer_anomalies = (
        sub2._infer_risk_level
    ) = static_failing_inference
    fb = await sub2.safe_concurrent_inference({"price": 50}, {})
    check("1.5 timeout fallback activates", fb.get("emergency") or fb.get("confidence_score", 0) >= 0.3)

    # 1.6 metrics
    s = sub.stats()
    check("1.6 metrics expose latency and cache", s["total_runs"] >= 42 and "avg_latency_ms" in s)


def static_failing_inference(data, context):
    raise TimeoutError("forced")


# ══════════════════════════════════════════════════════════════════════
# Phase 2: Intelligent Data Completion
# ══════════════════════════════════════════════════════════════════════
async def test_phase2() -> None:
    print("\n[Phase 2] Intelligent Data Completion")
    comp = IntelligentDataCompletion()

    # feed history
    for i in range(1, 21):
        comp.observe({"price": float(100 + i), "volume": float(500 + i * 5)})

    # 2.1 missing values ARE completed (original returned None)
    completed = await comp.complete_missing_data(
        {"price": None, "volume": None}, {}
    )
    check(
        "2.1 missing fields are completed",
        completed.get("price") is not None and completed.get("volume") is not None,
        f"price={completed.get('price'):.1f}, volume={completed.get('volume'):.1f}",
    )
    # 2.2 completion metadata present
    check(
        "2.2 completion report + rate present",
        "_completion" in completed and 0 < completed.get("_completion_rate", 0) <= 1,
        f"rate={completed.get('_completion_rate'):.0%}",
    )
    # 2.3 explicit values are never overwritten
    kept = await comp.complete_missing_data({"price": 123.4, "volume": None}, {})
    check("2.3 explicit values untouched", kept["price"] == 123.4)

    # 2.4 contextual hints strategy
    empty = IntelligentDataCompletion()
    hinted = await empty.complete_missing_data(
        {"price": None}, {"defaults": {"price": 99.0}, "hint_confidence": 0.6}
    )
    check("2.4 contextual hint strategy", hinted["price"] == 99.0)

    # 2.5 correlation strategy
    comp.set_correlation("volume", "price", 0.9)
    corr = await comp.complete_missing_data({"volume": 600.0, "price": None}, {})
    check(
        "2.5 correlation inference",
        isinstance(corr.get("price"), float) and corr.get("price", 0) > 0,
        f"price={corr.get('price'):.1f}",
    )

    # 2.6 unresolved field marked honestly
    empty = IntelligentDataCompletion()
    unresolved = await empty.complete_missing_data({"x": None}, {})
    check("2.6 unresolved fields reported honestly", unresolved.get("x") is None)


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Advanced Recovery Center
# ══════════════════════════════════════════════════════════════════════
async def test_phase3() -> None:
    print("\n[Phase 3] Advanced Recovery Center")
    rc = AdvancedRecoveryCenter()
    rc.register_component("agent", metadata={"related": ["memory", "kb"]})
    rc.register_component("memory")
    rc.register_component("kb")

    # 3.1 light error heals with tier-1 strategy
    ok1 = await rc.attempt_comprehensive_recovery("agent", {"type": "slow_cache_miss"})
    check("3.1 light error heals (tier 1)", ok1)

    # 3.2 timeout error escalates and still heals
    ok2 = await rc.attempt_comprehensive_recovery(
        "agent", {"type": "timeout", "message": "operation took too long"}
    )
    check("3.2 timeout escalates and heals", ok2)

    # 3.3 critical error uses critical tier
    ok3 = await rc.attempt_comprehensive_recovery(
        "agent", {"type": "critical", "message": "state corrupted"}
    )
    check("3.3 critical tier activates", ok3)

    # 3.4 cross-component recovery when related components exist
    ok4 = await rc.attempt_comprehensive_recovery(
        "agent", {"type": "cross_failure", "message": "related nodes hung"}
    )
    log = [r for r in rc.recovery_log if r["component"] == "agent"]
    xcomp = any(r["strategy"] == "cross_component_recovery" for r in log)
    check("3.4 cross-component strategy used", ok4 and xcomp)

    # 3.5 learning loop re-ranks strategies after history
    for _ in range(5):
        await rc.attempt_comprehensive_recovery("agent", {"type": "transient_warn"})
    rc.update_learning()
    check(
        "3.5 learning re-ranks strategies",
        "transient_warn" in rc.learning and len(rc.learning["transient_warn"]) > 0,
        f"signatures={len(rc.learning)}",
    )

    # 3.6 stats
    st = rc.stats()
    check(
        "3.6 stats + all 12 strategies registered",
        st["strategies_registered"] == 12 and st["success_rate"] >= 0.9,
        f"success_rate={st['success_rate']:.0%}",
    )


# ══════════════════════════════════════════════════════════════════════
# Phase 4: Multi-Layer Verification (10 layers)
# ══════════════════════════════════════════════════════════════════════
async def test_phase4() -> None:
    print("\n[Phase 4] Multi-Layer Verification (10 layers)")
    ver = MultiLayerVerification()

    # 4.1 good decision passes all layers
    good = {
        "decision_id": "dec-001",
        "action": "buy",
        "reasoning": "price dropped below support with high volume",
        "confidence": 0.8,
    }
    r1 = await ver.comprehensive_verification(good)
    check("4.1 healthy decision passes all 10 layers", r1["all_passed"], f"rate={r1['pass_rate']:.0%}")

    # 4.2 exactly 10 layers reported
    check("4.2 ten layers executed", r1["total_layers"] == 10 and len(r1["details"]) == 10)

    # 4.3 schema rejection (missing action)
    bad = {"decision_id": "dec-002", "reasoning": "no action field"}
    r2 = await ver.comprehensive_verification(bad)
    check("4.3 schema violation blocked", not r2["all_passed"] and r2.get("blocked_by") == "schema")

    # 4.4 invalid action blocked
    weird = {"decision_id": "dec-003", "action": "teleport", "reasoning": "some reasoning text"}
    r3 = await ver.comprehensive_verification(weird)
    check("4.4 invalid action blocked", not r3["all_passed"])

    # 4.5 confidence out of range fails confidence layer
    conf_bad = {
        "decision_id": "dec-004",
        "action": "sell",
        "reasoning": "momentum reversed strongly",
        "confidence": 5.0,
    }
    r4 = await ver.comprehensive_verification(conf_bad)
    check(
        "4.5 out-of-range confidence fails",
        not r4["details"]["confidence_threshold"]["passed"],
    )

    # 4.6 policy blocking (ethics/policy layer)
    policy_decision = {
        "decision_id": "dec-005",
        "action": "execute",
        "reasoning": "run blocked operation",
        "confidence": 0.8,
        "_policy": {"blocked_patterns": ["blocked operation"]},
    }
    r5 = await ver.comprehensive_verification(policy_decision)
    check("4.6 user policy blocks forbidden actions", not r5["details"]["ethics_check"]["passed"])

    # 4.7 anomaly detection fires on extreme confidence
    anomaly = {
        "decision_id": "dec-006",
        "action": "hold",
        "reasoning": "waiting for clearer signal",
        "confidence": 0.02,
    }
    r6 = await ver.comprehensive_verification(anomaly)
    check("4.7 anomaly layer flags extreme confidence", not r6["details"]["anomaly_detection"]["passed"])


# ══════════════════════════════════════════════════════════════════════
# Phase 5: Predictive Model (ensemble)
# ══════════════════════════════════════════════════════════════════════
async def test_phase5() -> None:
    print("\n[Phase 5] Enhanced Predictive Model")
    pm = PredictiveModel()

    # feed a rising series: 10, 20, ..., 100
    for i in range(1, 11):
        pm.observe(float(i * 10))

    # 5.1 ensemble responds to data (prediction > series mean ≈ 55)
    pred = await pm.predict_with_ensemble(
        {"target": None}, {"patterns": [[10, 20, 30, 40, 50]]}
    )
    check(
        "5.1 ensemble prediction reacts to data",
        isinstance(pred.get("prediction"), float) and pred["prediction"] > 50,
        f"prediction={pred['prediction']:.1f}",
    )
    # 5.2 all 4 sub-models contribute (pattern needs a prototype supplied)
    check(
        "5.2 four sub-models active",
        pred.get("sources") == 4 and len(pred.get("weights_used", {})) == 4,
    )
    # 5.3 insufficient data degrades gracefully
    pm2 = PredictiveModel()
    pred2 = await pm2.predict_with_ensemble({"x": 1}, {})
    check(
        "5.3 graceful degradation without data",
        pred2.get("confidence", 0) < 0.5 and pred2.get("note") == "insufficient_data",
    )
    # 5.4 calibration shifts weights after outcomes
    # mean/trend ar small errors, pattern deliberately wrong -> its weight must drop
    for _ in range(16):
        pm.record_outcome("mean", 60.0, 60.5)
        pm.record_outcome("trend", 60.0, 60.3)
        pm.record_outcome("ar", 60.0, 60.8)
        pm.record_outcome("pattern", 60.0, 200.0)
    check(
        "5.4 calibration rebalances weights",
        pm.calibrations >= 1
        and abs(sum(pm.weights.values()) - 1.0) < 1e-9
        and pm.weights.get("pattern", 0.25) < 0.25,
        f"weights={pm.weights}",
    )


# ══════════════════════════════════════════════════════════════════════
# Unified Engine
# ══════════════════════════════════════════════════════════════════════
async def test_engine() -> None:
    print("\n[Engine] HermesAIEngine (unified pipeline)")
    engine = HermesAIEngine()

    # feed observations so predictor works
    for i in range(1, 11):
        engine.completion.observe({"target": float(i * 10), "price": float(100 + i)})

    # 6.1 full pipeline passes on healthy input
    res = await engine.decide(
        {"decision_id": "eng-1", "action": "buy", "price": None},
        {"market": "bullish"},
    )
    check("6.1 full pipeline passes", res.get("passed") is True, f"latency={res.get('latency_ms')}ms")

    # 6.2 pipeline includes all 5 stages output
    check(
        "6.2 all stage outputs present",
        all(k in res for k in ("completion", "inference", "prediction", "verification")),
    )

    # 6.3 engine stats aggregate subsystem stats
    st = engine.stats()
    check(
        "6.3 aggregated stats",
        all(k in st for k in ("inference", "data_completion", "recovery_center",
                              "verification", "predictive_model"))
        and st["pipeline_runs"] >= 1,
    )

    # 6.4 engine self-heals when a stage raises
    engine.predictor.predict_with_ensemble = static_failing_predict
    res2 = await engine.decide(
        {"decision_id": "eng-2", "action": "sell", "price": 150}, {}
    )
    check(
        "6.4 engine self-heals on stage failure",
        res2.get("recovered") is True,
        f"error={res2.get('error')[:60]}",
    )
    engine.predictor.predict_with_ensemble = PredictiveModel.predict_with_ensemble.__get__(
        engine.predictor, PredictiveModel
    )

    # 6.5 concurrent pipeline runs stay consistent
    results = await asyncio.gather(
        *(engine.decide({"action": "hold", "price": 100 + i}, {}) for i in range(20))
    )
    check("6.5 20 concurrent pipeline runs consistent", all(isinstance(r, dict) for r in results))


def static_failing_predict(self, data, context):
    raise RuntimeError("simulated stage failure")


# ══════════════════════════════════════════════════════════════════════
async def main() -> None:
    print("=" * 70)
    print("HERMES-AI Comprehensive Test Suite (5 phases + unified engine)")
    print("=" * 70)
    await test_phase1()
    await test_phase2()
    await test_phase3()
    await test_phase4()
    await test_phase5()
    await test_engine()

    print("\n" + "=" * 70)
    print(f"TOTAL: {PASSED + FAILED} | PASSED: {PASSED} | FAILED: {FAILED}")
    print("=" * 70)

    # save report
    report_path = HERMES_ROOT / "tests" / "hermes_ai" / "hermes_ai_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": PASSED + FAILED,
                "passed": PASSED,
                "failed": FAILED,
                "results": REPORT,
                "suite_version": "2.0.0",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Report saved: {report_path}")
    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
