"""Offline fallback classifier: deterministic, zero-cost, keyword-scored."""

from decision_harness.heuristic import OfflineFallbackClassifier
from decision_harness.types import Tier


def test_mechanical_task_routes_fast():
    c = OfflineFallbackClassifier()
    for task in ("fix a typo in the footer", "rename this variable", "add an aria-label to the button"):
        assert c.classify(task).tier == Tier.FAST, task


def test_architectural_task_routes_performance():
    c = OfflineFallbackClassifier()
    for task in (
        "Design a zero-downtime migration with rollback plans",
        "Review the security architecture of our auth system",
    ):
        assert c.classify(task).tier == Tier.PERFORMANCE, task


def test_ordinary_work_routes_balanced():
    c = OfflineFallbackClassifier()
    assert c.classify("Implement the checkout summary endpoint with validation").tier == Tier.BALANCED


def test_classification_shape_and_markers():
    c = OfflineFallbackClassifier()
    cl = c.classify("fix a typo")
    assert 0.0 <= cl.tier_confidence <= 1.0
    assert 0.0 <= cl.complexity <= 3.0
    assert 0.0 <= cl.needs_planning <= 1.0
    assert cl.classifier == "heuristic"
    assert cl.latency_ms >= 0
