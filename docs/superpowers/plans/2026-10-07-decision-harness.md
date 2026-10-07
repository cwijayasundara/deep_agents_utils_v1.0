# decision_harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Production framework library (`decision_harness`) that puts OpenAI Decisions API-backed model selection and tool selection inside agent harnesses as LangChain middleware.

**Architecture:** Core (httpx-only) decides; the harness executes. `DecisionsClient` is the single hardened wire client (timeout/retry/breaker); `ModelSelector`/`ToolSelector` turn typed answers into routing decisions; two `AgentMiddleware` subclasses swap models/tools per request with sticky per-thread state and graceful degradation. Ported from the live-validated `model_router_v1.0` prototype.

**Tech Stack:** Python ≥3.10, httpx, pytest; langchain/deepagents as the `deepagents` extra (lazy-imported).

**Spec:** `SPEC.md` (this repo).

## Global Constraints

- TDD: no production code without a failing test first (RED → GREEN → commit).
- Core imports httpx only; langchain imports are lazy (middlewares only).
- 100% offline tests via `httpx.MockTransport`; no API keys needed.
- Names/semantics follow SPEC.md exactly; port validated logic from `model_router_v1.0` (src/ + tests/) rather than reinventing.

## File Structure

```
pyproject.toml                      # hatchling, decision_harness, extras deepagents/dev
src/decision_harness/
  __init__.py                       # public API exports
  errors.py                         # ConfigError, SelectorUnavailable
  types.py                          # Tier, ModelSpec, TaskClassification, RouteDecision, ToolSelection, BreakerConfig
  catalog.py                        # ModelCatalog + default catalog data (JSON)
  heuristic.py                      # OfflineFallbackClassifier
  ledger.py                         # CostLedger, UsageEntry
  decisions.py                      # DecisionsClient (sync+async): payload, parse, retry, breaker
  selector.py                       # ModelSelector (+LRU cache, fallback)
  tools.py                          # ToolSelector
  middleware/__init__.py
  middleware/model_selection.py     # ModelSelectionMiddleware
  middleware/tool_selection.py      # ToolSelectionMiddleware
  py.typed
tests/                              # mirror the above, offline only
examples/                           # basic_select.py, deepagents_demo.py
README.md, LICENSE, .gitignore, ci.yml
```

Reference code lives at `/Users/chamindawijayasundara/Documents/learning_101/model_router_v1.0` (read-only; port, don't import).

## Tasks

### Task 1 — Scaffold + types + errors
- [x] Write `tests/test_types.py`: Tier ordering (next/strongest), ModelSpec.label/cost/supports, SelectorUnavailable retryable flag
- [x] RED: run, verify failure
- [x] GREEN: `pyproject.toml`, `src/decision_harness/{__init__ (minimal),errors.py,types.py,py.typed}`, `.gitignore`, `conftest.py` (src/ first on sys.path)
- [x] Run tests, commit

### Task 2 — Catalog
- [x] Write `tests/test_catalog.py` (port from model_router tests/test_catalog.py): load defaults, merge overlay, by_tier cheapest-first, require/ConfigError, six providers present
- [x] RED
- [x] GREEN: `catalog.py` + `data/default_catalog.json` (three tier models + per-provider alternates, PLACEHOLDER prices)
- [x] Run, commit

### Task 3 — Heuristic fallback + ledger
- [x] Write `tests/test_heuristic.py` (port HeuristicClassifier tests) and `tests/test_ledger.py` (port CostLedger tests)
- [x] RED
- [x] GREEN: `heuristic.py`, `ledger.py`
- [x] Run, commit

### Task 4 — DecisionsClient (sync)
- [x] Write `tests/test_decisions.py`: classify payload (choice+score+predicate, guard, ZDR), select_tools payload (N predicates), response parsing (both question sets), refusal → SelectorUnavailable, malformed → SelectorUnavailable, retry on 429/5xx honoring Retry-After, no retry on 4xx, timeout → SelectorUnavailable
- [x] RED
- [x] GREEN: `decisions.py` sync half (payload builders, parse, retry loop, BreakerConfig dataclass)
- [x] Run, commit

### Task 5 — DecisionsClient (async + circuit breaker)
- [x] Write tests: `apost` parity; breaker: N consecutive failures → open (no network, SelectorUnavailable), cooldown expiry → half-open recovery, success resets
- [x] RED
- [x] GREEN: async half + breaker state machine (shared by sync/async)
- [x] Run, commit

### Task 6 — ModelSelector + policy
- [x] Write `tests/test_policy.py` (port TierPolicy tests incl. mechanical-task rule + retry escalation) and `tests/test_selector.py`: one-request classify → RouteDecision, LRU cache hit (client called once), fallback to heuristic on SelectorUnavailable (event recorded), no fallback on programmer error
- [x] RED
- [x] GREEN: `policy.py`, `selector.py`
- [x] Run, commit

### Task 7 — ToolSelector
- [x] Write `tests/test_tools.py` (port DecisionToolSelector tests to Decisions-only): payload, parse, threshold, max_tools, always_include, on_none all/top1/none, failure mapping
- [x] RED
- [x] GREEN: `tools.py`
- [x] Run, commit

### Task 8 — ModelSelectionMiddleware
- [x] Write `tests/test_model_middleware.py` (port from model_router tests/test_langchain_integration.py, adapted): spec→chat model mapping (fireworks/baseten/openrouter→openai-compat base_url; anthropic/google native; reasoning_effort pinning), first-call swap, sticky state, ledger with latency event, escalation on failure, async parity
- [x] RED
- [x] GREEN: `middleware/model_selection.py` (+ lazy langchain import shim)
- [x] Run, commit

### Task 9 — ToolSelectionMiddleware
- [x] Write `tests/test_tool_middleware.py`: per-call filter, sticky state, stale-cache re-select, degrade-to-all on SelectorUnavailable, no-tools/no-text passthrough, always_include/max_tools/on_none pass-through, async
- [x] RED
- [x] GREEN: `middleware/tool_selection.py`
- [x] Run, commit

### Task 10 — deepagents end-to-end (offline)
- [x] Write `tests/test_deepagents_e2e.py`: create_deep_agent + both middlewares + fake models + tmp workspace → routed model used, tools filtered, edit-and-test completes
- [x] RED
- [x] GREEN: fix whatever the integration surfaces
- [x] Run, commit

### Task 11 — Public API, examples, README, CI
- [x] `__init__.py` full exports; `examples/basic_select.py` (offline-selectable), `examples/deepagents_demo.py`
- [x] README (quickstart, catalog overlay, production checklist), LICENSE (MIT), `.github/workflows/ci.yml` (pytest 3.10/3.12/3.14 + ruff)
- [x] Full suite green + ruff clean; final commit
