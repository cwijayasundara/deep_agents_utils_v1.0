"""Model catalog: defaults across providers, overlay merging, tier queries."""

import pytest

from decision_harness import ConfigError, ModelCatalog, ModelSpec, Tier


@pytest.fixture
def catalog():
    return ModelCatalog.load()


def test_defaults_have_all_three_tiers(catalog):
    for tier in Tier.order():
        specs = catalog.by_tier(tier)
        assert specs, f"no {tier.value} models in defaults"


def test_primary_tier_models_from_the_blog(catalog):
    assert catalog.require("accounts/fireworks/models/glm-5p3-flash").tier == Tier.FAST
    assert catalog.require("gpt-6.1-sol").tier == Tier.BALANCED
    assert catalog.require("gpt-6-astra").tier == Tier.PERFORMANCE


def test_alternates_across_six_providers(catalog):
    providers = {spec.provider for spec in catalog.all()}
    assert {"fireworks", "openai", "anthropic", "google", "baseten", "openrouter"} <= providers


def test_by_tier_cheapest_first(catalog):
    for tier in Tier.order():
        specs = catalog.by_tier(tier)
        costs = [s.expected_cost() for s in specs]
        assert costs == sorted(costs)


def test_classifier_entries_are_not_routable_chat_models(catalog):
    for spec in catalog.by_tier(Tier.FAST):
        assert spec.id  # chat models only selected via by_tier default kind
    assert all(spec.kind == "chat" for tier in Tier.order() for spec in catalog.by_tier(tier))


def test_overlay_merges_over_defaults(tmp_path):
    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        ModelCatalog.from_dict(
            {
                "models": [
                    {
                        "id": "my-model",
                        "provider": "mine",
                        "tier": "fast",
                        "chat_provider": "openai",
                        "input_price": 0.0,
                        "output_price": 0.0,
                    }
                ]
            }
        ).to_json()
        if hasattr(ModelCatalog, "to_json")
        else __import__("json").dumps(
            {
                "models": [
                    {"id": "my-model", "provider": "mine", "tier": "fast", "chat_provider": "openai"}
                ]
            }
        )
    )
    catalog = ModelCatalog.load(path=overlay)
    assert catalog.get("my-model") is not None
    assert catalog.get("gpt-6-astra") is not None  # defaults still there


def test_overlay_overrides_existing_entry(tmp_path):
    import json

    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "gpt-6.1-sol",
                        "provider": "openai",
                        "tier": "balanced",
                        "reasoning_effort": "high",  # user pins differently
                    }
                ]
            }
        )
    )
    catalog = ModelCatalog.load(path=overlay)
    assert catalog.require("gpt-6.1-sol").reasoning_effort == "high"


def test_unknown_tier_in_entry_raises():
    with pytest.raises(ConfigError):
        ModelCatalog.from_dict({"models": [{"id": "x", "provider": "p", "tier": "cosmic"}]})


def test_require_missing_raises():
    with pytest.raises(ConfigError):
        ModelCatalog().require("nope")
