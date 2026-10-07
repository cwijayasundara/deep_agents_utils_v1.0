"""Core types: tiers, model specs, decisions, and error semantics."""

import pytest

from decision_harness.errors import ConfigError, SelectorUnavailable
from decision_harness.types import ModelSpec, RouteDecision, Tier


def test_tier_ordering():
    assert Tier.next(Tier.FAST) == Tier.BALANCED
    assert Tier.next(Tier.PERFORMANCE) == Tier.PERFORMANCE  # ceiling
    assert Tier.strongest(Tier.FAST, Tier.PERFORMANCE) == Tier.PERFORMANCE
    assert Tier.strongest(Tier.BALANCED, Tier.FAST) == Tier.BALANCED


def test_model_spec_label_key_support():
    spec = ModelSpec(
        id="gpt-6.1-sol",
        provider="openai",
        tier=Tier.BALANCED,
        input_price=1.75,
        output_price=14.0,
        capabilities=frozenset({"tools", "vision"}),
    )
    assert spec.label() == "openai/gpt-6.1-sol"
    assert spec.key == ("openai", "gpt-6.1-sol")
    assert spec.supports("tools") and not spec.supports("audio")
    assert spec.cost(1_000_000, 1_000_000) == pytest.approx(15.75)


def test_selector_unavailable_is_not_config_error():
    err = SelectorUnavailable("decisions down", retryable=True)
    assert err.retryable
    assert not SelectorUnavailable("bad payload").retryable
    assert issubclass(SelectorUnavailable, Exception)
    assert issubclass(ConfigError, Exception)


def test_route_decision_defaults():
    spec = ModelSpec(id="m", provider="p")
    d = RouteDecision(model=spec, tier=spec.tier)
    assert d.events == [] and d.sticky is False and d.reason == "classified"
