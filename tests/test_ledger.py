"""Cost ledger: per-call usage entries, totals, JSON export."""

import json

import pytest

from decision_harness.ledger import CostLedger


def test_record_and_totals():
    ledger = CostLedger()
    ledger.record(
        provider="fireworks", model="glm", input_tokens=100, output_tokens=50,
        cost=0.0001, latency_ms=900, session="t1", reason="classified", events=["e1"],
    )
    ledger.record(
        provider="openai", model="sol", input_tokens=10, output_tokens=5,
        cost=0.5, latency_ms=1200, session="t2",
    )
    totals = ledger.totals()
    assert totals["grand"]["calls"] == 2
    assert totals["grand"]["cost"] == pytest.approx(0.5001)
    assert totals["grand"]["input_tokens"] == 110
    assert totals["by_model"]["fireworks/glm"]["calls"] == 1


def test_session_and_events_preserved():
    ledger = CostLedger()
    entry = ledger.record(
        provider="p", model="m", input_tokens=1, output_tokens=1,
        cost=0.0, latency_ms=5, session="thread-9", reason="classified", events=["a", "b"],
    )
    assert entry.session == "thread-9"
    assert entry.events == ["a", "b"]
    assert ledger.entries[0].ts > 0


def test_empty_totals():
    totals = CostLedger().totals()
    assert totals["grand"]["calls"] == 0 and totals["grand"]["cost"] == 0


def test_to_json_roundtrip(tmp_path):
    ledger = CostLedger()
    ledger.record(provider="p", model="m", input_tokens=1, output_tokens=1,
                  cost=0.01, latency_ms=5, session="s")
    path = tmp_path / "ledger.json"
    path.write_text(ledger.to_json())
    data = json.loads(path.read_text())
    assert data["totals"]["grand"]["calls"] == 1
    assert data["entries"][0]["model"] == "m"
