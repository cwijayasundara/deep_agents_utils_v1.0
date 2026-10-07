"""EscalationPolicy: conservative overrides incl. the mechanical-task rule."""

import pytest

from decision_harness import ModelCatalog, TaskClassification, Tier
from decision_harness.policy import EscalationPolicy, TierFloors


def make_cls(tier=Tier.FAST, conf=0.9, complexity=0.0, planning=0.0, caps=frozenset(), est_in=1_000):
    return TaskClassification(
        tier=tier, tier_confidence=conf, complexity=complexity, needs_planning=planning,
        required_capabilities=caps, est_input_tokens=est_in,
    )


@pytest.fixture
def catalog():
    return ModelCatalog.load()


def test_clean_classification_stays_in_tier(catalog):
    d = EscalationPolicy().decide(make_cls(), catalog)
    assert d.model.tier == Tier.FAST
    assert d.reason == "classified"
    assert d.events == []


def test_uncertainty_escalates_for_non_mechanical_work(catalog):
    d = EscalationPolicy().decide(make_cls(conf=0.5, complexity=1.0), catalog)
    assert d.model.tier == Tier.PERFORMANCE
    assert any("uncertain" in e for e in d.events)


def test_mechanical_task_with_uncertain_route_stays_in_tier(catalog):
    d = EscalationPolicy().decide(make_cls(conf=0.6, complexity=0.04, planning=0.48), catalog)
    assert d.model.tier == Tier.FAST
    assert d.reason == "classified"
    assert any("mechanical" in e for e in d.events)


def test_complexity_escalates(catalog):
    d = EscalationPolicy().decide(make_cls(tier=Tier.BALANCED, conf=0.9, complexity=2.0), catalog)
    assert d.model.tier == Tier.PERFORMANCE


def test_planning_probability_escalates(catalog):
    d = EscalationPolicy().decide(make_cls(complexity=1.0, planning=0.8), catalog)
    assert d.model.tier == Tier.PERFORMANCE


def test_retry_escalates_one_tier(catalog):
    d = EscalationPolicy().decide(make_cls(), catalog, reason="retry")
    assert d.model.tier == Tier.BALANCED


def test_capability_shortfall_escalates_tier(catalog):
    d = EscalationPolicy().decide(make_cls(caps=frozenset({"vision"})), catalog)
    assert d.model.tier != Tier.FAST


def test_huge_context_falls_back_to_strongest(catalog):
    d = EscalationPolicy().decide(make_cls(est_in=10_000_000), catalog)
    assert d.model.intelligence >= max(s.intelligence for s in catalog.all())


def test_tier_floors_are_configurable(catalog):
    policy = EscalationPolicy(floors=TierFloors(route_confidence_floor=0.3))
    d = policy.decide(make_cls(conf=0.4, complexity=1.0), catalog)
    assert d.model.tier == Tier.FAST  # 0.4 >= 0.3 -> no escalation
