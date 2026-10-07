"""ToolSelectionMiddleware: filter tools per request, degrade to all, sticky option."""

import asyncio

import pytest

pytest.importorskip("langchain")

from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command

from decision_harness.errors import SelectorUnavailable
from decision_harness.middleware.tool_selection import ToolSelectionMiddleware
from decision_harness.types import ToolSelection


class StubToolSelector:
    def __init__(self, result):
        self.result = result  # list[str] | Exception
        self.calls = 0

    def select(self, task, tools, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return ToolSelection(selected=list(self.result), probabilities={}, latency_ms=5)

    async def aselect(self, task, tools, **kwargs):
        return self.select(task, tools, **kwargs)


def _t(name, desc):
    @tool(name, description=desc)
    def _f() -> str:
        """ok"""
    return _f


TOOLS = [_t("read_file", "read"), _t("edit_file", "edit"), _t("deploy", "deploy")]


def _request(model=None, tools=TOOLS, state=None):
    return ModelRequest(
        model=model or GenericFakeChatModel(messages=iter([AIMessage("ok")] * 4)),
        messages=[HumanMessage("fix a typo in the footer")],
        tools=list(tools),
        state=state or {},
    )


def _handler(captured=None):
    def handler(request):
        if captured is not None:
            captured["tools"] = list(request.tools)
        return ModelResponse(result=[AIMessage("ok")])

    return handler


def test_filters_tools_per_call():
    selector = StubToolSelector(["read_file"])
    mw = ToolSelectionMiddleware(selector=selector)
    captured: dict = {}
    mw.wrap_model_call(_request(), _handler(captured))
    assert [t.name for t in captured["tools"]] == ["read_file"]


def test_per_call_reselects_every_model_call():
    selector = StubToolSelector(["read_file"])
    mw = ToolSelectionMiddleware(selector=selector, per_call=True)
    state: dict = {}
    mw.wrap_model_call(_request(state=state), _handler())
    mw.wrap_model_call(_request(state=state), _handler())
    assert selector.calls == 2


def test_sticky_mode_selects_once_and_persists():
    selector = StubToolSelector(["read_file"])
    mw = ToolSelectionMiddleware(selector=selector, per_call=False)
    state: dict = {}
    out = mw.wrap_model_call(_request(state=state), _handler())
    assert isinstance(out, ExtendedModelResponse)
    state.update(out.command.update)  # the graph applies the command
    mw.wrap_model_call(_request(state=state), _handler())
    assert selector.calls == 1


def test_sticky_stale_selection_reroutes():
    selector = StubToolSelector(["edit_file"])
    mw = ToolSelectionMiddleware(selector=selector, per_call=False)
    state = {"selected_tools": ["ghost_tool"]}
    mw.wrap_model_call(_request(state=state), _handler())
    assert selector.calls == 1  # stale -> re-selected


def test_degrades_to_all_tools_on_failure():
    selector = StubToolSelector(SelectorUnavailable("down"))
    mw = ToolSelectionMiddleware(selector=selector)
    captured: dict = {}
    mw.wrap_model_call(_request(), _handler(captured))
    assert {t.name for t in captured["tools"]} == {"read_file", "edit_file", "deploy"}
    assert any("keeping all tools" in e for e in mw.last.events)


def test_no_tools_is_passthrough():
    selector = StubToolSelector(["x"])
    mw = ToolSelectionMiddleware(selector=selector)
    captured: dict = {}
    mw.wrap_model_call(_request(tools=[]), _handler(captured))
    assert selector.calls == 0
    assert captured["tools"] == []


def test_no_human_message_is_passthrough():
    selector = StubToolSelector(["read_file"])
    mw = ToolSelectionMiddleware(selector=selector)
    captured: dict = {}
    req = ModelRequest(
        model=GenericFakeChatModel(messages=iter([AIMessage("ok")] * 4)),
        messages=[AIMessage("assistant greeting")],
        tools=list(TOOLS),
        state={},
    )
    mw.wrap_model_call(req, _handler(captured))
    assert selector.calls == 0
    assert len(captured["tools"]) == 3


def test_kwargs_pass_through():
    selector = StubToolSelector(["read_file"])
    mw = ToolSelectionMiddleware(selector=selector, always_include=["deploy"], max_tools=2, on_none="top1")
    captured: dict = {}
    seen_kwargs = {}
    original = selector.select

    def spy(task, tools, **kwargs):
        seen_kwargs.update(kwargs)
        return original(task, tools, **kwargs)

    selector.select = spy
    mw.wrap_model_call(_request(), _handler(captured))
    assert seen_kwargs["always_include"] == ["deploy"]
    assert seen_kwargs["max_tools"] == 2
    assert seen_kwargs["on_none"] == "top1"
    assert {t.name for t in captured["tools"]} == {"read_file", "deploy"}


def test_async_matches_sync():
    selector = StubToolSelector(["read_file"])
    mw = ToolSelectionMiddleware(selector=selector)
    captured: dict = {}

    async def ahandler(request):
        captured["tools"] = list(request.tools)
        return ModelResponse(result=[AIMessage("ok")])

    out = asyncio.run(mw.awrap_model_call(_request(), ahandler))
    assert [t.name for t in captured["tools"]] == ["read_file"]
