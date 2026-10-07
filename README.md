# decision_harness

Production decision-making utilities for agent harnesses, built on the
**OpenAI Decisions API**. The two expensive per-request decisions an agent
harness makes — *which model* and *which tools* — become one fast typed-questions
request each, with calibrated probabilities, circuit breaking, and graceful
degradation.

Implements the architecture from LangChain's [*How to Build a Model Router in
the Harness*](https://www.langchain.com/blog/how-to-build-a-model-router-in-the-harness/)
as drop-in LangChain/deepagents middleware.

```bash
pip install ".[deepagents]"
```

## Quickstart

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from decision_harness.decisions import DecisionsClient
from decision_harness.middleware.model_selection import ModelSelectionMiddleware
from decision_harness.middleware.tool_selection import ToolSelectionMiddleware
from decision_harness.selector import ModelSelector
from decision_harness.tools import ToolSelector

client = DecisionsClient()  # reads DECISIONS_API_KEY / OPENAI_API_KEY
agent = create_deep_agent(
    model="openai:gpt-6-astra",  # default & escalation ceiling
    middleware=[
        ToolSelectionMiddleware(selector=ToolSelector(client=client)),
        ModelSelectionMiddleware(selector=ModelSelector(client=client)),
    ],
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": [{"role": "user", "content": "fix a typo in the footer"}]},
    config={"configurable": {"thread_id": "user-123"}},
)
```

- The thread's **first** human message is classified (tier choice +
  probability, complexity score, planning predicate — one request) and the
  model is swapped to the routed tier. Every later call on the same
  `thread_id` stays on that model (prompt cache stays warm).
- Tool selection runs one predicate question per tool in a single request;
  tools below threshold are filtered. If Decisions is down, the agent keeps
  all its tools.
- Every call lands in the ledger with tokens, cost, latency, and the full
  decision audit trail.

## Catalog

Defaults ship the three Pareto-frontier tier models from the blog post plus
alternates for Fireworks, OpenAI, Anthropic, Google, BaseTen, and OpenRouter.
**All prices are placeholders** — verify against provider pricing pages, or
overlay your own:

```python
catalog = ModelCatalog.load(path="my_catalog.json")  # merged over defaults
```

## Production checklist

- Verify placeholder prices/context windows in the catalog.
- Size the circuit breaker for your traffic (`breaker_consecutive_failures`,
  `breaker_cooldown_s` on `DecisionsClient`).
- Tune `TierFloors` on `EscalationPolicy` for your task mix.
- Wire `ModelSelectionMiddleware(on_decision=...)` to your observability stack.

## License

MIT
