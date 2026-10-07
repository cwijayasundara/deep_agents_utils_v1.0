# SPEC: `decision_harness`

Production decision-making utilities for agent harnesses, built on the **OpenAI
Decisions API**. Two decisions every agent harness makes per request — *which
model* and *which tools* — answered by one typed-questions backend instead of
full LLM calls.

Validated design: this packages what `model_router_v1.0` proved live (sticky
per-thread routing, ~83× cost cut on mechanical tasks, 350 ms tool selection
over 10 tools with calibrated probabilities) as framework-only code.

---

## Goals

1. `ModelSelector` — classify a task (tier choice + probability, complexity
   score, planning predicate) in one Decisions API request and route it to the
   cheapest capable model through a conservative escalation policy.
2. `ModelSelectionMiddleware` — LangChain/deepagents middleware that swaps the
   model an agent calls, sticky per thread, with cost tracking and
   failure escalation.
3. `ToolSelectionMiddleware` — middleware that filters the agent's tool set per
   request, one predicate question per tool, calibrated probabilities.
4. Production guarantees: bounded worst-case latency, circuit breaker, retry
   with backoff, graceful degradation that never breaks the agent.

## Non-goals

- No OpenRouter / Jev / gateway backends — OpenAI Decisions API only.
- No direct model dispatch (no chat-completions client); harnesses infer via
  LangChain `BaseChatModel`. The library decides; the harness executes.
- No eval harness / traffic mining / A-B scaffolding.

---

## Module layout

```
src/decision_harness/
├── decisions.py       # DecisionsClient + AsyncDecisionsClient (wire protocol)
├── catalog.py         # ModelCatalog: tier models across 6 providers, overlay support
├── policy.py          # EscalationPolicy: conservative tier overrides
├── selector.py        # ModelSelector: task -> RouteDecision
├── tools.py           # ToolSelector: task + tools -> ToolSelection
├── heuristic.py       # OfflineFallbackClassifier (degradation path)
├── ledger.py          # CostLedger / UsageEntry
├── errors.py          # SelectorUnavailable, ConfigError
├── types.py           # Tier, ModelSpec, TaskClassification, RouteDecision, ToolSelection, ...
├── middleware/
│   ├── model_selection.py   # ModelSelectionMiddleware
│   └── tool_selection.py    # ToolSelectionMiddleware
└── py.typed
```

Dependency policy: `httpx` only at core; `langchain`/`deepagents` are the
`deepagents` extra. Middlewares import langchain lazily so the core stays
installable anywhere.

---

## 1. `decisions.py` — DecisionsClient

One client, two question sets:

- **classify**: `choice` (tier, with probabilities) + `score` (complexity 0–3)
  + `predicate` (needs-planning) — one `POST /v1/decisions`.
- **select_tools**: N `predicate` questions, one per tool — one `POST /v1/decisions`.

```python
class DecisionsClient:
    def __init__(self, *, model="gpt-6-luna", api_key=None, timeout=5.0,
                 max_retries=2, breaker: BreakerConfig | None = None,
                 zero_data_retention=True, transport=None): ...
    def post(self, payload: dict) -> dict: ...          # sync
    async def apost(self, payload: dict) -> dict: ...   # async, shared pool
```

Production behaviour:

| Concern | Behaviour |
|---|---|
| Timeout | 5 s default (read+connect), sync and async |
| Retry | 429/5xx: exponential backoff with jitter, honours `Retry-After`; `max_retries` bounds it (default 2) |
| Circuit breaker | `breaker.consecutive_failures` (default 3) failures → open for `cooldown_s` (default 30) → `SelectorUnavailable` raised immediately without network |
| Response validation | Missing/refused answers → `SelectorUnavailable` (typed, message names the question) |
| ZDR | `zero_data_retention=True` sends ZDR / no-prompt-training flags |
| Injection guard | Task text and tool descriptions are sent as data; the guard sentence rides in every question payload |
| Observability | Every request/response outcome recorded with latency, model, question count |

Env: `DECISIONS_API_KEY` (or `OPENAI_API_KEY`), optional
`DECISIONS_BASE_URL` for testing/proxies.

## 2. `catalog.py` — ModelCatalog

Data-driven registry; everything editable as data. Defaults ship the blog's
three Pareto-frontier tier models as primary targets, **plus alternates per
provider** so deployments can route to whichever vendor they have keys for:

| Tier | Primary (default) | Alternates in defaults |
|---|---|---|
| fast | `accounts/fireworks/models/glm-5p3-flash` (xhigh) | openai, google, baseten, openrouter fast models |
| balanced | `gpt-6.1-sol` (medium) | anthropic, google, baseten, openrouter balanced models |
| performance | `gpt-6-astra` (low) | anthropic, google, baseten, openrouter performance models |

- Prices/context windows/intelligence scores are **PLACEHOLDERS**, marked in
  the data file; verify before production (per the blog).
- `ModelCatalog.load(path=...)` merges a user overlay over defaults.
- Catalog entries carry the LangChain provider mapping (`openai-chat` with
  `base_url` for OpenAI-compatible vendors — fireworks, baseten, openrouter;
  native `anthropic` / `google-genai` entries).

## 3. `policy.py` — EscalationPolicy

Ported and live-validated semantics:

- Route to the classified tier; escalate to `override_target` (default
  performance) when: route confidence < 0.68 **and the task is not clearly
  mechanical** (the live-validated rule — complexity < 1.0 and planning <
  floor stay in tier), complexity ≥ 1.75, planning ≥ 0.6.
- Escalate on capability shortfall or estimated-context overflow.
- `reason="retry"` escalates one tier after a provider failure.
- Floors configurable via `TierFloors`.

## 4. `selector.py` — ModelSelector

```python
class ModelSelector:
    def __init__(self, client, catalog=None, policy=None, cache_size=256): ...
    def select(self, task: str) -> RouteDecision: ...
    async def aselect(self, task: str) -> RouteDecision: ...
```

- One Decisions request → typed classification → policy → `RouteDecision`
  (model spec, tier, events audit trail, classification with latency).
- LRU cache on task text (default 256 entries) — repeated identical tasks skip
  the round trip.
- Circuit-breaker open or Decisions failure → `heuristic.OfflineFallbackClassifier`
  (deterministic, zero-cost) with a `classifier_fallback` event. The selector
  never raises to the caller except on programmer error (empty task).

## 5. `tools.py` — ToolSelector

```python
class ToolSelector:
    def __init__(self, client, *, threshold=0.5): ...
    def select(self, task, tools: list[tuple[str, str]], *, max_tools=None,
               always_include=None, on_none="all") -> ToolSelection: ...
    async def aselect(...) -> ToolSelection: ...
```

- One predicate question per tool, one request; answers are calibrated
  probabilities.
- Threshold selection ranked by probability; `max_tools` cap; `always_include`
  bypasses the threshold; `on_none`: `"all"` (default — when in doubt,
  include) / `"top1"` / `"none"`.
- Decisions failure → `SelectorUnavailable` (the middleware owns degradation).

## 6. Middlewares (`middleware/`)

Both are `AgentMiddleware` subclasses with sync + async hooks, sharing the
langchain-only details (state schema, `ModelRequest.override`, `Command`
updates, GraphBubbleUp passthrough).

**`ModelSelectionMiddleware`**

- First model call of a thread: classify the last human message via
  `ModelSelector`, swap `request.model` (catalog spec → `init_chat_model` with
  the entry's pinned reasoning effort; provider mapping per catalog).
- Decision persisted in state (`routed_model`); with a checkpointer the thread
  stays on that model (prompt-cache friendly).
- Provider failure → policy escalation (`reason="retry"`), `max_retries`
  bound (default 2).
- Every successful call → `CostLedger` (usage from `usage_metadata`,
  session = thread id, events = audit trail incl. classification latency).
- Optional `on_decision` callback hook for external observability.

**`ToolSelectionMiddleware`**

- Per model call (default `per_call=True`; tool needs change mid-loop) or
  sticky per thread (`per_call=False`, state-persisted `selected_tools`).
- `always_include`, `max_tools`, `on_none` pass through to the selector.
- Degradation: selector unavailable/no human message → **keep all tools**;
  stale cached selection (tool gone) → re-select.
- Last selection exposed as `middleware.last` for observability.

## 7. Errors and types

`errors.py`: `ConfigError`, `SelectorUnavailable` (with `.retryable`).
`types.py`: `Tier`, `ModelSpec`, `TaskClassification`, `RouteDecision`,
`ToolSelection`, `BreakerConfig` — dataclasses, all serializable.

## 8. Testing (100% offline)

- `httpx.MockTransport` for every wire interaction — no API keys, no network.
- Unit: client payload/response/retry/breaker; catalog merge; policy table
  (incl. mechanical rule); selector cache/fallback; tool threshold/cap/fallback.
- Middleware: model swap, sticky state, escalation, ledger, degrade-to-all —
  via `create_agent` + fake chat models; one end-to-end through
  `create_deep_agent`.
- Examples: runnable offline (`examples/` mirrors the four flows).

## 9. Packaging & CI

- Name: **`decision_harness`** (repo `deep_agents_utils_v1.0`), MIT, Python ≥ 3.10.
- `[project.optional-dependencies] deepagents = ["deepagents>=0.7.22"]`, `dev = ["pytest>=8", "ruff"]`.
- GitHub Actions: pytest (3.10/3.12/3.14) + ruff on push/PR.
- README: quickstart, catalog overlay guide, production checklist (verify
  placeholder prices, breaker sizing, cache sizing).
