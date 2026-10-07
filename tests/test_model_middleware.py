"""ModelSelectionMiddleware: swap the model, sticky per thread, ledger, escalation."""

import asyncio

import pytest

pytest.importorskip("langchain")
pytest.importorskip("deepagents")  # middleware pulls langchain-agents machinery via deepagents extra

from langchain.agents import create_agent
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from decision_harness import ModelCatalog, Tier
from decision_harness.middleware.model_selection import ModelSelectionMiddleware
from decision_harness.selector import ModelSelector

FAST = ModelCatalog.load().require("accounts/fireworks/routers/glm-5p3-fast")


def classify_body(tier="fast", conf=0.9, complexity=0.2, planning=0.1):
    return {
        "answers": [
            {"name": "tier", "type": "choice", "choice": tier,
             "probabilities": [{"value": tier, "probability": conf}]},
            {"name": "complexity", "type": "score", "score": complexity},
            {"name": "needs_planning", "type": "predicate", "probability": planning},
        ]
    }


class RecordingModel(GenericFakeChatModel):
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RaisingModel(GenericFakeChatModel):
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        raise RuntimeError("simulated provider outage")


def _fake(text="ok"):
    # Distinct AIMessage objects: langchain mutates ids on generation.
    return RecordingModel(messages=iter([AIMessage(text) for _ in range(8)]))


def make_mw(tier="fast", conf=0.9, models=None, **client_kwargs):
    def handler(request):
        return httpx.Response(200, json=classify_body(tier=tier, conf=conf))

    mock = httpx.MockTransport(handler)
    client = DecisionsClient(api_key="k", transport=mock, async_transport=mock, **client_kwargs)
    selector = ModelSelector(client=client, catalog=ModelCatalog.load())
    return ModelSelectionMiddleware(
        selector=selector, models=models or {}, fallback_model="openai:gpt-6-astra"
    )


import httpx

from decision_harness.decisions import DecisionsClient


def _request(model, state=None):
    return ModelRequest(
        model=model, messages=[HumanMessage("fix a typo in the footer")], state=state or {}
    )


def _ok_handler(captured=None):
    def handler(request):
        if captured is not None:
            captured["model"] = request.model
        reply = request.model.invoke(list(request.messages))
        return ModelResponse(result=[reply])

    return handler


# --------------------------------------------------------------------------
# spec -> chat model adapter
# --------------------------------------------------------------------------


def test_spec_to_chat_model_maps_openai(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "decision_harness.middleware.model_selection.init_chat_model",
        lambda *a, **k: captured.update(model=a[0] if a else k.get("model"), kwargs=k) or object(),
    )
    from decision_harness.middleware.model_selection import spec_to_chat_model

    spec_to_chat_model(ModelCatalog.load().require("gpt-6.1-sol"))
    assert captured["model"] == "gpt-6.1-sol"
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["reasoning_effort"] == "medium"


def test_spec_to_chat_model_openai_compat_vendors_get_base_url(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "decision_harness.middleware.model_selection.init_chat_model",
        lambda *a, **k: captured.update(kwargs=k) or object(),
    )
    import os

    from decision_harness.middleware.model_selection import spec_to_chat_model

    os.environ["FIREWORKS_API_KEY"] = "fw-test"
    try:
        spec_to_chat_model(FAST)
    finally:
        del os.environ["FIREWORKS_API_KEY"]
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert captured["kwargs"]["api_key"] == "fw-test"
    assert captured["kwargs"]["reasoning_effort"] == FAST.reasoning_effort


# --------------------------------------------------------------------------
# Middleware behaviour
# --------------------------------------------------------------------------


def test_first_call_swaps_model_and_persists():
    fast_model = _fake()
    mw = make_mw(models={FAST.label(): fast_model})
    strong = _fake()
    out = mw.wrap_model_call(_request(strong), _ok_handler())
    assert isinstance(out, ExtendedModelResponse)
    assert out.command.update["routed_model"] == FAST.label()
    assert fast_model.calls == 1  # handler invoked the routed model


def test_sticky_from_state_no_reclassification():
    fast_model = _fake()
    mw = make_mw(models={FAST.label(): fast_model})
    captured: dict = {}
    state = {"routed_model": FAST.label()}
    mw.wrap_model_call(_request(_fake(), state=state), _ok_handler(captured))
    assert captured["model"] is fast_model
    # selector not touched: cached state, no HTTP. (verify via ledger entries)
    assert len(mw.ledger.entries) == 1
    assert all("classified in" not in e for e in mw.ledger.entries[0].events)


def test_ledger_records_usage_and_classification_latency():
    usage_msg = AIMessage("ok", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    mw = make_mw(models={FAST.label(): RecordingModel(messages=iter([usage_msg] * 4))})
    mw.wrap_model_call(_request(_fake()), _ok_handler())
    entry = mw.ledger.entries[0]
    assert entry.input_tokens == 10 and entry.output_tokens == 5
    assert entry.provider == "fireworks"
    assert entry.cost == pytest.approx(FAST.cost(10, 5))
    assert any("classified in" in e for e in entry.events)


def test_escalates_after_provider_failure():
    flaky = RaisingModel(messages=iter([AIMessage("never")]))
    catalog = ModelCatalog.load()
    # Policy picks the cheapest capable balanced model on retry — could be any
    # provider; inject fakes for all of them so construction never dials out.
    models = {FAST.label(): flaky}
    for tier in (Tier.BALANCED, Tier.PERFORMANCE):
        for spec in catalog.by_tier(tier):
            models[spec.label()] = _fake("recovered")
    mw = make_mw(models=models)
    captured: dict = {}
    out = mw.wrap_model_call(_request(_fake()), _ok_handler(captured))
    assert captured["model"].calls == 1  # the escalated model answered
    assert flaky.calls == 1
    assert out.model_response.result[0].content == "recovered"
    assert len(mw.ledger.entries) == 1  # only the successful call recorded
    assert any("failed" in e for e in mw.ledger.entries[0].events)


def test_async_wrap_model_call():
    fast_model = _fake()
    mw = make_mw(models={FAST.label(): fast_model})

    async def ahandler(request):
        return ModelResponse(result=[request.model.invoke(list(request.messages))])

    out = asyncio.run(mw.awrap_model_call(_request(_fake()), ahandler))
    assert out.command.update["routed_model"] == FAST.label()
    assert len(mw.ledger.entries) == 1


# --------------------------------------------------------------------------
# End-to-end through create_agent
# --------------------------------------------------------------------------


def test_agent_routes_and_stays_sticky():
    fast = _fake("fast says ok")
    strong = _fake("strong")
    mw = make_mw(models={FAST.label(): fast})
    agent = create_agent(model=strong, middleware=[mw], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    r1 = agent.invoke({"messages": [HumanMessage("fix a typo")]}, config=cfg)
    r2 = agent.invoke({"messages": [HumanMessage("second turn")]}, config=cfg)
    assert r1["messages"][-1].content == "fast says ok"
    assert r2["messages"][-1].content == "fast says ok"
    assert strong.calls == 0
    assert mw.selector._cache  # classified exactly once, cached after
