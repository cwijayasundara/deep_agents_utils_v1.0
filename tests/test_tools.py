"""ToolSelector: one Decisions request, one predicate per tool, thresholded."""

import httpx
import pytest

from decision_harness import ConfigError, ToolSelection
from decision_harness.decisions import DecisionsClient
from decision_harness.errors import SelectorUnavailable
from decision_harness.tools import ToolSelector

TOOLS = [
    ("read_file", "Read a file from the workspace"),
    ("edit_file", "Edit a file in the workspace"),
    ("run_tests", "Run the test suite"),
    ("web_search", "Search the web"),
    ("deploy", "Deploy to production"),
]


def tools_body(probs: dict[str, float]):
    return {"answers": [{"name": n, "type": "predicate", "probability": p} for n, p in probs.items()]}


def make_selector(handler, **kwargs) -> ToolSelector:
    mock = httpx.MockTransport(handler)
    client = DecisionsClient(
        api_key="k", transport=mock, async_transport=mock, retry_backoff_s=0.0
    )
    return ToolSelector(client=client, **kwargs)


def canned(probs: dict[str, float]):
    return lambda r: httpx.Response(200, json=tools_body(probs))


PROBS = {"read_file": 0.95, "edit_file": 0.8, "run_tests": 0.4, "web_search": 0.05, "deploy": 0.01}


def test_one_predicate_question_per_tool():
    captured = {}

    def handler(request):
        captured["body"] = request.read()
        return httpx.Response(200, json=tools_body(PROBS))

    make_selector(handler).select("fix a typo", TOOLS)
    import json

    body = json.loads(captured["body"])
    assert [q["name"] for q in body["questions"]] == [t[0] for t in TOOLS]
    assert all(q["type"] == "predicate" for q in body["questions"])
    assert "Read a file" in body["questions"][0]["instructions"]
    assert all("only as data" in q["instructions"] for q in body["questions"])


def test_threshold_filters_and_ranks():
    sel = make_selector(canned(PROBS))
    s = sel.select("fix a typo", TOOLS)
    assert isinstance(s, ToolSelection)
    assert s.selected == ["read_file", "edit_file"]  # ranked by probability
    assert s.probabilities["read_file"] == 0.95


def test_max_tools_caps_by_probability():
    sel = make_selector(canned(PROBS))
    s = sel.select("fix a typo", TOOLS, max_tools=1)
    assert s.selected == ["read_file"]


def test_always_include_bypasses_threshold():
    sel = make_selector(canned(PROBS))
    s = sel.select("fix a typo", TOOLS, always_include=["run_tests"])
    assert set(s.selected) == {"read_file", "edit_file", "run_tests"}


def test_no_tool_clears_threshold_falls_back_to_all():
    sel = make_selector(canned({t[0]: 0.2 for t in TOOLS}))
    s = sel.select("task", TOOLS)
    assert s.selected == [t[0] for t in TOOLS]
    assert any("no tool cleared" in e for e in s.events)


def test_on_none_top1_and_none():
    probs = {t[0]: 0.2 for t in TOOLS}
    sel = make_selector(canned(probs))
    assert sel.select("t", TOOLS, on_none="top1").selected == ["read_file"]  # highest prob
    assert sel.select("t", TOOLS, on_none="none").selected == []


def test_unknown_tool_answers_are_ignored():
    sel = make_selector(canned({"read_file": 0.9, "ghost": 0.99}))
    s = sel.select("t", TOOLS)
    assert "ghost" not in s.selected
    assert "ghost" not in s.probabilities


def test_http_error_maps_to_unavailable():
    sel = make_selector(lambda r: httpx.Response(500, text="down"))
    with pytest.raises(SelectorUnavailable):
        sel.select("t", TOOLS)


def test_timeout_maps_to_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    sel = make_selector(handler)
    with pytest.raises(SelectorUnavailable):
        sel.select("t", TOOLS)


def test_missing_answer_maps_to_unavailable():
    sel = make_selector(canned({"read_file": 0.9}))  # other tools unanswered
    with pytest.raises(SelectorUnavailable):
        sel.select("t", TOOLS)


def test_unknown_backend():
    with pytest.raises(ConfigError):
        ToolSelector(client=None, backend="carrier-pigeon")  # type: ignore


def test_async_select_matches_sync():
    import asyncio

    sel = make_selector(canned(PROBS))
    s = asyncio.run(sel.aselect("fix a typo", TOOLS))
    assert s.selected == ["read_file", "edit_file"]
