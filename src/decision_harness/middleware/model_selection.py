"""ModelSelectionMiddleware: route the model an agent calls, sticky per thread.

The blog's semantics, production-hardened: classify the thread's first human
message with the Decisions API, swap the model, persist the decision in state
(checkpointer keeps the thread on it), escalate on provider failure, and
record every call in the ledger.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..errors import ConfigError
from ..ledger import CostLedger
from ..types import ModelSpec, RouteDecision, Tier

try:  # langchain is an optional extra; keep core importable without it
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
    )
    from langchain.chat_models import init_chat_model
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage, HumanMessage
    from langgraph.errors import GraphBubbleUp
    from langgraph.types import Command
    from typing_extensions import NotRequired

    _LANGCHAIN = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _LANGCHAIN = False

if _LANGCHAIN:

    class _State(AgentState[Any]):
        routed_model: NotRequired[str]

    # Providers speaking the OpenAI-compatible chat API behind a different host.
    _OPENAI_COMPAT = {"fireworks", "baseten", "openrouter"}

    def spec_to_chat_model(spec: ModelSpec, **overrides: Any) -> BaseChatModel:
        """Catalog entry -> LangChain chat model (pinned reasoning effort included)."""
        params: dict[str, Any] = {"model": spec.id, "model_provider": spec.chat_provider}
        if spec.base_url:
            params["base_url"] = spec.base_url
        if spec.api_key_env:
            key = __import__("os").environ.get(spec.api_key_env)
            if key:
                params["api_key"] = key
        if spec.reasoning_effort and spec.chat_provider == "openai":
            params["reasoning_effort"] = spec.reasoning_effort
        params.update(overrides)
        return init_chat_model(**params)

    class ModelSelectionMiddleware(AgentMiddleware):
        """Swap the model an agent calls per the routed tier decision.

        Args:
            selector: a `ModelSelector` (Decisions-backed; fallback built in).
            fallback_model: model spec used when the catalog can't satisfy a
                decision (default "openai:gpt-6-astra" via init_chat_model).
            models: explicit chat models by catalog label (`spec.label()`),
                bypassing `init_chat_model` — how tests inject fakes.
            max_retries: failed-call escalations before giving up (default 2).
        """

        state_schema = _State

        def __init__(
            self,
            *,
            selector,
            fallback_model: str = "openai:gpt-6-astra",
            models: dict[str, BaseChatModel] | None = None,
            max_retries: int = 2,
            on_decision: Callable[[RouteDecision], None] | None = None,
        ) -> None:
            self.selector = selector
            self.fallback_model = fallback_model
            self._injected = dict(models or {})
            self.max_retries = max_retries
            self.on_decision = on_decision
            self.ledger = CostLedger()
            self._chat_models: dict[str, BaseChatModel] = {}

        # -- routing -----------------------------------------------------------

        def _routed_from_state(self, state: Any) -> RouteDecision | None:
            if not state:
                return None
            try:
                label = state.get("routed_model")
            except AttributeError:
                label = getattr(state, "routed_model", None)
            if not label or "/" not in label:
                return None
            provider, model_id = label.split("/", 1)
            spec = self.selector.catalog.get(model_id, provider)
            if spec is None:
                return None  # stale/foreign id -> re-route
            return RouteDecision(
                model=spec,
                tier=spec.tier,
                reason="sticky",
                events=[f"sticky: thread stays on {spec.label()}"],
                sticky=True,
            )

        def _decide(self, request: ModelRequest) -> RouteDecision | None:
            routed = self._routed_from_state(request.state)
            if routed is not None:
                if self.on_decision:
                    self.on_decision(routed)
                return routed
            text = self._last_human_text(request.messages)
            if not text:
                return None
            decision = self.selector.select(text)
            if decision.classification:
                decision.events.append(
                    f"classified in {decision.classification.latency_ms}ms "
                    f"by {decision.classification.classifier}"
                )
            return decision

        @staticmethod
        def _last_human_text(messages: list[BaseMessage] | None) -> str:
            for m in reversed(messages or []):
                if isinstance(m, HumanMessage):
                    content = m.content
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
                        return " ".join(p for p in parts if p)
            return ""

        # -- model resolution / escalation --------------------------------------

        def _model_for(self, spec: ModelSpec) -> BaseChatModel:
            label = spec.label()
            if label not in self._chat_models:
                injected = self._injected.get(label)
                if injected is not None:
                    self._chat_models[label] = injected
                elif spec.chat_provider in _OPENAI_COMPAT:
                    self._chat_models[label] = spec_to_chat_model(spec)
                else:
                    self._chat_models[label] = spec_to_chat_model(spec)
            return self._chat_models[label]

        def _next_on_failure(self, failed: ModelSpec, decision: RouteDecision) -> RouteDecision:
            if decision.classification is not None:
                return self.selector.policy.decide(
                    decision.classification, self.selector.catalog, reason="retry"
                )
            for tier in Tier.order()[Tier.order().index(failed.tier) + 1 :]:
                for candidate in self.selector.catalog.by_tier(tier):
                    if candidate.key != failed.key and failed.capabilities <= candidate.capabilities:
                        return RouteDecision(
                            model=candidate,
                            tier=tier,
                            reason="retry",
                            events=[f"escalated to {candidate.label()} after failure"],
                        )
            strongest = self.selector.catalog.strongest()
            return RouteDecision(
                model=strongest,
                tier=strongest.tier,
                reason="retry",
                events=[f"escalated to strongest model {strongest.label()}"],
            )

        # -- bookkeeping ---------------------------------------------------------

        @staticmethod
        def _usage(response: ModelResponse) -> tuple[int, int]:
            input_tokens = output_tokens = 0
            for message in response.result:
                usage = getattr(message, "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0) or 0
                output_tokens += usage.get("output_tokens", 0) or 0
            return input_tokens, output_tokens

        @staticmethod
        def _thread_id(request: ModelRequest) -> str | None:
            runtime = request.runtime
            config = getattr(runtime, "config", None) or {}
            configurable = config.get("configurable") or {}
            thread_id = configurable.get("thread_id")
            return str(thread_id) if thread_id else None

        def _record(self, spec: ModelSpec, events: list[str], reason: str, latency_ms: int, usage) -> None:
            input_tokens, output_tokens = usage
            self.ledger.record(
                provider=spec.provider,
                model=spec.id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=spec.cost(input_tokens, output_tokens),
                latency_ms=latency_ms,
                session=None,
                reason=reason,
                events=events,
            )

        # -- harness hooks ---------------------------------------------------------

        def wrap_model_call(self, request: ModelRequest, handler):
            decision = self._decide(request)
            if decision is None:
                return handler(request)

            spec = decision.model
            events = list(decision.events)
            request = request.override(model=self._model_for(spec))

            attempt = 0
            started = time.perf_counter()
            while True:
                try:
                    response = handler(request)
                    break
                except GraphBubbleUp:
                    raise
                except Exception as e:
                    attempt += 1
                    events.append(f"attempt {attempt}: {spec.label()} failed ({e})")
                    if attempt > self.max_retries:
                        raise
                    decision = self._next_on_failure(spec, decision)
                    spec = decision.model
                    events.append(f"retry: switching to {spec.label()}")
                    request = request.override(model=self._model_for(spec))

            self._record(
                spec, events, decision.reason,
                int((time.perf_counter() - started) * 1000), self._usage(response),
            )
            if self.on_decision:
                self.on_decision(decision)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"routed_model": spec.label()}),
            )

        async def awrap_model_call(self, request: ModelRequest, handler):
            decision = await self._adecide(request)
            if decision is None:
                return await handler(request)

            spec = decision.model
            events = list(decision.events)
            request = request.override(model=self._model_for(spec))

            attempt = 0
            started = time.perf_counter()
            while True:
                try:
                    response = await handler(request)
                    break
                except GraphBubbleUp:
                    raise
                except Exception as e:
                    attempt += 1
                    events.append(f"attempt {attempt}: {spec.label()} failed ({e})")
                    if attempt > self.max_retries:
                        raise
                    decision = self._next_on_failure(spec, decision)
                    spec = decision.model
                    events.append(f"retry: switching to {spec.label()}")
                    request = request.override(model=self._model_for(spec))

            self._record(
                spec, events, decision.reason,
                int((time.perf_counter() - started) * 1000), self._usage(response),
            )
            if self.on_decision:
                self.on_decision(decision)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"routed_model": spec.label()}),
            )

        async def _adecide(self, request: ModelRequest) -> RouteDecision | None:
            routed = self._routed_from_state(request.state)
            if routed is not None:
                if self.on_decision:
                    self.on_decision(routed)
                return routed
            text = self._last_human_text(request.messages)
            if not text:
                return None
            return await self.selector.aselect(text)

else:  # pragma: no cover - placeholder keeps the module importable

    class ModelSelectionMiddleware:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ConfigError("langchain is required: pip install 'decision_harness[deepagents]'")
