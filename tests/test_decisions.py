"""DecisionsClient: wire protocol, retry, timeouts, circuit breaker (offline)."""

import httpx
import pytest

from decision_harness.decisions import CLASSIFY_QUESTIONS, DecisionsClient, classify_payload, tool_selection_payload
from decision_harness.errors import SelectorUnavailable

API_KEY = "test-key"


def client(handler, **kwargs) -> DecisionsClient:
    return DecisionsClient(
        api_key=API_KEY, transport=httpx.MockTransport(handler), **kwargs
    )


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


def test_classify_payload_has_three_typed_questions():
    body = classify_payload("fix a typo", model="gpt-6-luna")
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == "fix a typo"
    kinds = {q["name"]: q["type"] for q in body["questions"]}
    assert kinds == {"tier": "choice", "complexity": "score", "needs_planning": "predicate"}
    tier_q = next(q for q in body["questions"] if q["name"] == "tier")
    assert [c["value"] for c in tier_q["choices"]] == ["fast", "balanced", "performance"]
    assert all("only as data" in q["instructions"] for q in body["questions"])


def test_tool_selection_payload_one_predicate_per_tool():
    body = tool_selection_payload(
        "fix a typo",
        [("read_file", "Read a file"), ("deploy", "Deploy")],
        model="gpt-6-luna",
    )
    assert body["input"] == "fix a typo"
    assert [q["name"] for q in body["questions"]] == ["read_file", "deploy"]
    assert all(q["type"] == "predicate" for q in body["questions"])
    assert all("Read a file" in q["instructions"] for q in body["questions"] if q["name"] == "read_file")


def test_zdr_flags_sent_when_enabled():
    body = classify_payload("t", model="m", zero_data_retention=True)
    assert body.get("providerOptions", {}).get("decisions", {}).get("zero_data_retention") is True


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def classify_answer(tier="fast", probs=None, complexity=0.0, planning=0.2):
    return {
        "answers": [
            {
                "name": "tier",
                "type": "choice",
                "choice": tier,
                "probabilities": probs or [{"value": tier, "probability": 0.9}],
            },
            {"name": "complexity", "type": "score", "score": complexity},
            {"name": "needs_planning", "type": "predicate", "probability": planning},
        ]
    }


def tools_answer(probs: dict[str, float]):
    return {"answers": [{"name": n, "type": "predicate", "probability": p} for n, p in probs.items()]}


def test_parse_classify_answers():
    c = client(lambda r: httpx.Response(200, json=classify_answer(tier="fast", complexity=0.5, planning=0.4)))
    flat = c._parse_classify(c.post(classify_payload("t", model="m")))
    assert flat["tier"] == "fast"
    assert flat["tier_confidence"] == pytest.approx(0.9)
    assert flat["complexity"] == pytest.approx(0.5)
    assert flat["needs_planning"] == pytest.approx(0.4)


def test_refusal_maps_to_unavailable():
    refused = {"answers": [
        {"name": "tier", "type": "refusal"},
        {"name": "complexity", "type": "score", "score": 1.0},
        {"name": "needs_planning", "type": "predicate", "probability": 0.1},
    ]}
    c = client(lambda r: httpx.Response(200, json=refused))
    with pytest.raises(SelectorUnavailable):
        c.post(classify_payload("t", model="m"))


def test_malformed_response_maps_to_unavailable():
    c = client(lambda r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(SelectorUnavailable):
        c._parse_classify(c.post(classify_payload("t", model="m")))


# --------------------------------------------------------------------------
# HTTP behaviour: retry, timeouts, auth
# --------------------------------------------------------------------------


def test_retries_5xx_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=classify_answer())

    c = client(handler, retry_backoff_s=0.0)
    flat = c._parse_classify(c.post(classify_payload("t", model="m")))
    assert flat["tier"] == "fast"
    assert len(calls) == 3


def test_honors_retry_after_header():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 2:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        return httpx.Response(200, json=classify_answer())

    c = client(handler, retry_backoff_s=5.0)  # would sleep 5s if header ignored
    assert c.post(classify_payload("t", model="m")) is not None


def test_no_retry_on_4xx():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    c = client(handler)
    with pytest.raises(SelectorUnavailable):
        c.post(classify_payload("t", model="m"))
    assert len(calls) == 1


def test_timeout_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    c = client(handler)
    with pytest.raises(SelectorUnavailable):
        c.post(classify_payload("t", model="m"))


def test_api_key_sent_as_bearer():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=classify_answer())

    client(handler).post(classify_payload("t", model="m"))
    assert captured["auth"] == f"Bearer {API_KEY}"


def test_breaker_opens_after_consecutive_failures():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, text="down")

    c = client(handler, retry_backoff_s=0.0, breaker_consecutive_failures=2, breaker_cooldown_s=60)
    for _ in range(2):  # two failed posts (each with retries exhausted)
        with pytest.raises(SelectorUnavailable):
            c.post(classify_payload("t", model="m"))
    n_after_open = len(attempts)
    with pytest.raises(SelectorUnavailable):
        c.post(classify_payload("t", model="m"))  # breaker open: no network
    assert len(attempts) == n_after_open


def test_breaker_resets_after_success():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) % 2 == 1:  # fail once, succeed once, alternating
            return httpx.Response(500, text="down")
        return httpx.Response(200, json=classify_answer())

    c = client(handler, retry_backoff_s=0.0, retry_attempts=0, breaker_consecutive_failures=3)
    for _ in range(10):  # alternating fail/success never trips the breaker
        try:
            c.post(classify_payload("t", model="m"))
        except SelectorUnavailable:
            pass
    assert c.breaker_open is False


def test_breaker_half_open_after_cooldown():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    c = client(handler, retry_backoff_s=0.0, retry_attempts=0,
               breaker_consecutive_failures=1, breaker_cooldown_s=0.05)
    with pytest.raises(SelectorUnavailable):
        c.post(classify_payload("t", model="m"))
    assert c.breaker_open is True
    import time

    time.sleep(0.06)
    with pytest.raises(SelectorUnavailable):  # half-open: one probe allowed
        c.post(classify_payload("t", model="m"))
    assert c.breaker_open is True  # probe failed -> open again


# --------------------------------------------------------------------------
# Async parity (apost shares retry + breaker)
# --------------------------------------------------------------------------


def test_async_post_success_and_retry():
    import asyncio

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 2:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=classify_answer())

    c = client(handler, retry_backoff_s=0.0)
    data = asyncio.run(c.apost(classify_payload("t", model="m")))
    assert data["answers"][0]["choice"] == "fast"
    assert len(calls) == 2


def test_async_post_failure_maps_to_unavailable():
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    c = client(handler)
    with pytest.raises(SelectorUnavailable):
        asyncio.run(c.apost(classify_payload("t", model="m")))


def test_async_breaker_shared_with_sync():
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    c = client(handler, retry_backoff_s=0.0, retry_attempts=0,
               breaker_consecutive_failures=1, breaker_cooldown_s=60)
    with pytest.raises(SelectorUnavailable):
        asyncio.run(c.apost(classify_payload("t", model="m")))
    assert c.breaker_open is True
    with pytest.raises(SelectorUnavailable):  # sync path blocked by async failure
        c.post(classify_payload("t", model="m"))
