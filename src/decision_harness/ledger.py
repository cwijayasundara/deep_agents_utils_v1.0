"""Cost accounting: a per-call ledger with totals and JSON export."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class UsageEntry:
    ts: float
    session: str | None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int
    reason: str = ""
    events: list[str] = field(default_factory=list)


class CostLedger:
    """Records usage for every dispatched call. Feed totals into your reports."""

    def __init__(self) -> None:
        self.entries: list[UsageEntry] = []

    def record(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: int,
        session: str | None = None,
        reason: str = "",
        events: list[str] | None = None,
    ) -> UsageEntry:
        entry = UsageEntry(
            ts=time.time(),
            session=session,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            reason=reason,
            events=list(events or []),
        )
        self.entries.append(entry)
        return entry

    def totals(self) -> dict[str, Any]:
        by_model: dict[str, dict[str, float]] = {}
        for e in self.entries:
            m = by_model.setdefault(
                f"{e.provider}/{e.model}",
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            )
            m["calls"] += 1
            m["input_tokens"] += e.input_tokens
            m["output_tokens"] += e.output_tokens
            m["cost"] += e.cost
        grand = {
            "calls": len(self.entries),
            "input_tokens": sum(e.input_tokens for e in self.entries),
            "output_tokens": sum(e.output_tokens for e in self.entries),
            "cost": sum(e.cost for e in self.entries),
        }
        return {"by_model": by_model, "grand": grand}

    def to_json(self, path: str | None = None) -> str:
        payload = json.dumps(
            {"totals": self.totals(), "entries": [asdict(e) for e in self.entries]}, indent=2
        )
        if path:
            Path(path).write_text(payload)
        return payload
