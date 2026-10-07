"""ToolSelectionMiddleware: filter the agent's tools per request via Decisions.

Drop-in alternative to LangChain's `LLMToolSelectorMiddleware`: one fast
request with one predicate question per tool instead of a full LLM call.
A failed selector degrades to "keep all tools" — the agent never loses its
tools because of the router.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..errors import SelectorUnavailable
from ..types import ToolSelection

try:  # langchain is an optional extra; keep core importable without it
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ExtendedModelResponse,
        ModelRequest,
    )
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command
    from typing_extensions import NotRequired

    _LANGCHAIN = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _LANGCHAIN = False

if _LANGCHAIN:

    class _State(AgentState[Any]):
        selected_tools: NotRequired[list[str]]

    class ToolSelectionMiddleware(AgentMiddleware):
        """Filter the tools a model call sees, per a decision-model selection.

        Args:
            selector: a `ToolSelector` (Decisions-backed).
            always_include: tool names that bypass selection entirely.
            max_tools: cap on selected tools (highest probability wins).
            per_call: re-select on every model call (default; tool needs can
                change mid-loop). `False` selects once per thread, persisted
                in state like model routing.
            on_none: when no tool clears the threshold — "all" (default),
                "top1", or "none".
        """

        state_schema = _State

        def __init__(
            self,
            *,
            selector,
            always_include: list[str] | None = None,
            max_tools: int | None = None,
            per_call: bool = True,
            on_none: str = "all",
        ) -> None:
            self.selector = selector
            self.always_include = list(always_include or [])
            self.max_tools = max_tools
            self.per_call = per_call
            self.on_none = on_none
            self.last: ToolSelection | None = None  # last selection, for observability

        # -- helpers ------------------------------------------------------------

        @staticmethod
        def _last_human_text(messages: list[Any]) -> str:
            for m in reversed(messages or []):
                if isinstance(m, HumanMessage):
                    content = m.content
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
                        return " ".join(p for p in parts if p)
            return ""

        def _filter(self, request: ModelRequest, selected: list[str]) -> ModelRequest:
            keep = set(selected) | set(self.always_include)
            selected_tools = [t for t in request.tools if not isinstance(t, dict) and t.name in keep]
            provider_tools = [t for t in request.tools if isinstance(t, dict)]
            return request.override(tools=[*selected_tools, *provider_tools])

        def _select(self, request: ModelRequest) -> list[str] | None:
            """Selected tool names, or None to keep everything (no text / failure)."""
            text = self._last_human_text(request.messages)
            if not text or not request.tools:
                return None
            pairs = [(t.name, t.description or "") for t in request.tools if not isinstance(t, dict)]
            try:
                selection = self.selector.select(
                    text,
                    pairs,
                    max_tools=self.max_tools,
                    always_include=self.always_include,
                    on_none=self.on_none,
                )
            except SelectorUnavailable as e:
                self.last = ToolSelection(
                    selected=[], events=[f"tool selector unavailable, keeping all tools: {e}"]
                )
                return None
            self.last = selection
            return selection.selected

        def _cached_selection(self, request: ModelRequest) -> list[str] | None:
            if self.per_call or not request.state:
                return None
            try:
                cached = request.state.get("selected_tools")
            except AttributeError:
                cached = getattr(request.state, "selected_tools", None)
            if not cached:
                return None
            names = {t.name for t in request.tools if not isinstance(t, dict)}
            return list(cached) if set(cached) <= names else None  # stale -> re-select

        # -- harness hooks --------------------------------------------------------

        def wrap_model_call(self, request: ModelRequest, handler):
            if not request.tools:
                return handler(request)
            cached = self._cached_selection(request)
            if cached is not None:
                return handler(self._filter(request, cached))

            selected = self._select(request)
            if selected is None:
                return handler(request)
            filtered = self._filter(request, selected)
            if self.per_call:
                return handler(filtered)
            return ExtendedModelResponse(
                model_response=handler(filtered),
                command=Command(update={"selected_tools": selected}),
            )

        async def awrap_model_call(self, request: ModelRequest, handler):
            if not request.tools:
                return await handler(request)
            cached = self._cached_selection(request)
            if cached is not None:
                return await handler(self._filter(request, cached))

            selected = await asyncio.get_running_loop().run_in_executor(None, self._select, request)
            if selected is None:
                return await handler(request)
            filtered = self._filter(request, selected)
            if self.per_call:
                return await handler(filtered)
            return ExtendedModelResponse(
                model_response=await handler(filtered),
                command=Command(update={"selected_tools": selected}),
            )

else:  # pragma: no cover - placeholder keeps the module importable

    class ToolSelectionMiddleware:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            from ..errors import ConfigError

            raise ConfigError("langchain is required: pip install 'decision_harness[deepagents]'")
