"""ToolSelector: which tools does this task need? One request, N predicates.

Answers come back as calibrated probabilities; selection is a threshold with
conservative fallbacks. Any wire failure raises `SelectorUnavailable` - the
middleware owns degradation ("keep all tools").
"""

from __future__ import annotations

import time
from typing import Literal

from .decisions import DecisionsClient, tool_selection_payload
from .errors import ConfigError
from .types import ToolSelection

_ON_NONE = ("all", "none", "top1")


class ToolSelector:
    def __init__(
        self,
        *,
        client: DecisionsClient | None = None,
        threshold: float = 0.5,
        backend: str | None = None,  # reserved; Decisions API is the only backend
    ) -> None:
        if backend is not None and backend != "decisions":
            raise ConfigError(f"unknown tool-selector backend {backend!r}; only 'decisions' is supported")
        if client is None:
            client = DecisionsClient()
        self.client = client
        self.threshold = float(threshold)

    def select(
        self,
        task: str,
        tools: list[tuple[str, str]],
        *,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
        on_none: Literal["all", "none", "top1"] = "all",
    ) -> ToolSelection:
        return self._decide(self._select_via_decisions(task, tools), tools, max_tools, always_include, on_none)

    async def aselect(
        self,
        task: str,
        tools: list[tuple[str, str]],
        *,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
        on_none: Literal["all", "none", "top1"] = "all",
    ) -> ToolSelection:
        return self._decide(
            await self._aselect_via_decisions(task, tools), tools, max_tools, always_include, on_none
        )

    # -- internals ---------------------------------------------------------------

    def _select_via_decisions(self, task: str, tools: list[tuple[str, str]]) -> dict[str, float]:
        started = time.perf_counter()
        data = self.client.post(
            tool_selection_payload(task, tools, zero_data_retention=self.client.zero_data_retention)
        )
        probabilities = self.client._parse_tools(data, [t[0] for t in tools])
        self._elapsed = int((time.perf_counter() - started) * 1000)
        return probabilities

    async def _aselect_via_decisions(self, task: str, tools: list[tuple[str, str]]) -> dict[str, float]:
        started = time.perf_counter()
        data = await self.client.apost(
            tool_selection_payload(task, tools, zero_data_retention=self.client.zero_data_retention)
        )
        probabilities = self.client._parse_tools(data, [t[0] for t in tools])
        self._elapsed = int((time.perf_counter() - started) * 1000)
        return probabilities

    def _decide(
        self,
        probabilities: dict[str, float],
        tools: list[tuple[str, str]],
        max_tools: int | None,
        always_include: list[str] | None,
        on_none: str,
    ) -> ToolSelection:
        if on_none not in _ON_NONE:
            raise ConfigError(f"on_none must be one of {_ON_NONE}")
        events: list[str] = []
        always = set(always_include or [])
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)

        selected = [n for n, p in ranked if p >= self.threshold or n in always]
        if max_tools is not None:
            kept = [n for n in selected if n in always]
            kept += [n for n in selected if n not in always][: max(0, max_tools - len(kept))]
            selected = kept
        if not selected:
            if on_none == "all":
                selected = [t[0] for t in tools]
                events.append(
                    f"no tool cleared threshold {self.threshold:.2f} -> keeping all ({len(tools)}) tools"
                )
            elif on_none == "top1" and ranked:
                selected = [ranked[0][0]]
                events.append(f"no tool cleared threshold {self.threshold:.2f} -> keeping top-1 {ranked[0][0]}")
            else:
                events.append(f"no tool cleared threshold {self.threshold:.2f} -> no tools")

        return ToolSelection(
            selected=selected,
            probabilities=probabilities,
            events=events,
            latency_ms=getattr(self, "_elapsed", 0),
        )
