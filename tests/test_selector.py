"""ModelSelector: one Decisions request -> RouteDecision, with cache + fallback."""

import httpx
import pytest

from decision_harness import ModelCatalog, Tier
from decision_harness.decisions import DecisionsClient
from decision_harness.selector import ModelSelector

TASK = "Add an aria-label to the close button and run its test."


def classify_body(tier="fast", conf=0.9, complexity=0.2, planning=0.1):
    return {
        "answers": [
            {"name": "tier", "type": "choice", "choice": tier,
             "probabilities": [{"value": tier, "probability": conf}]},
            {"name": "complexity", "type": "score", "score": complexity},
            {"name": "needs_planning", "type": "predicate", "probability": planning},
        ]
    }


def make_selector(handler, **kwargs) -> tuple[ModelSelector, list]:
    calls = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return handler(request)

    mock = httpx.MockTransport(counting_handler)
    client = DecisionsClient(
        api_key="k", transport=mock, async_transport=mock, retry_backoff_s=0.0, **kwargs
    )
    selector = ModelSelector(client=client, catalog=ModelCatalog.load())
    return selector, calls


def test_select_returns_routed_decision():
    selector, calls = make_selector(lambda r: httpx.Response(200, json=classify_body(tier="fast", conf=0.9)))
    d = selector.select(TASK)
    assert d.model.tier == Tier.FAST
    assert d.model.label() == "fireworks/accounts/fireworks/models/glm-5p3-flash"
    assert d.classification.classifier.startswith("decisions:")
    assert d.classification.latency_ms >= 0
    assert len(calls) == 1


def test_low_complexity_classification_skips_policy_escalation():
    selector, _ = make_selector(lambda r: httpx.Response(200, json=classify_body(conf=0.71, complexity=0.04)))
    assert selector.select(TASK).reason == "classified"


def test_cache_hit_skips_the_round_trip():
    selector, calls = make_selector(lambda r: httpx.Response(200, json=classify_body()))
    d1 = selector.select(TASK)
    d2 = selector.select(TASK)
    assert len(calls) == 1  # LRU cache: identical task answered from memory
    assert d2.model.label() == d1.model.label()
    assert d2.classification.classifier.startswith("cache:")


def test_falls_back_to_heuristic_when_decisions_down():
    selector, calls = make_selector(lambda r: httpx.Response(500, text="down"), retry_attempts=0)
    d = selector.select(TASK)
    assert d.model.tier in Tier.order()  # still routed
    assert any("fallback" in e for e in d.events)
    assert d.classification.classifier == "heuristic"


def test_breaker_open_degrades_instantly():
    selector, calls = make_selector(
        lambda r: httpx.Response(500, text="down"),
        retry_attempts=0, breaker_consecutive_failures=1, breaker_cooldown_s=60,
    )
    selector.select(TASK)  # trips the breaker
    n = len(calls)
    selector.select("another task entirely")  # different text -> cache miss
    assert len(calls) == n  # no network: breaker open -> heuristic directly


def test_async_select_matches_sync():
    import asyncio

    selector, calls = make_selector(lambda r: httpx.Response(200, json=classify_body(tier="performance", conf=0.95)))
    d = asyncio.run(selector.aselect(TASK))
    assert d.model.tier == Tier.PERFORMANCE
    assert len(calls) == 1
