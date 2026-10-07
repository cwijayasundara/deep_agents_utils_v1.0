"""End-to-end through the real deepagents harness (offline, fake models)."""

import pytest

pytest.importorskip("deepagents")

import httpx
from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from decision_harness import ModelCatalog
from decision_harness.decisions import DecisionsClient
from decision_harness.middleware.model_selection import ModelSelectionMiddleware
from decision_harness.middleware.tool_selection import ToolSelectionMiddleware
from decision_harness.selector import ModelSelector
from decision_harness.tools import ToolSelector

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


class FakeWithBind(GenericFakeChatModel):
    """deepagents binds tools on the routed model; fakes just ignore them."""

    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _fake(text):
    return FakeWithBind(messages=iter([AIMessage(text) for _ in range(12)]))


def build_agent(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=classify_body())

    mock = httpx.MockTransport(handler)
    client = DecisionsClient(api_key="k", transport=mock, async_transport=mock)

    catalog = ModelCatalog.load()
    routed = _fake("fast model handled it")

    model_mw = ModelSelectionMiddleware(
        selector=ModelSelector(client=client, catalog=catalog),
        models={FAST.label(): routed},
    )
    tool_mw = ToolSelectionMiddleware(
        selector=ToolSelector(client=client),
        always_include=["read_file"],
    )
    agent = create_deep_agent(
        model=_fake("should not be used"),
        middleware=[tool_mw, model_mw],
        backend=LocalShellBackend(root_dir=tmp_path),
        checkpointer=InMemorySaver(),
    )
    return agent, model_mw, tool_mw, routed


def test_deepagents_routes_and_executes(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    agent, model_mw, tool_mw, routed = build_agent(tmp_path)

    result = agent.invoke(
        {"messages": [HumanMessage("fix a typo")]},
        config={"configurable": {"thread_id": "e2e-1"}},
    )
    assert result["messages"][-1].content == "fast model handled it"
    assert routed.calls >= 1
    assert model_mw.ledger.totals()["grand"]["calls"] >= 1
    assert tool_mw.last is not None  # tool selection ran
