"""Full deepagents harness: routed model + selected tools + real files.

    python examples/deepagents_demo.py [workspace] ["task"]

Needs DECISIONS_API_KEY (or OPENAI_API_KEY) plus a provider key for whichever
model gets routed (e.g. FIREWORKS_API_KEY for the fast tier). WARNING:
LocalShellBackend executes commands on your machine - point it at a
workspace you trust.
"""

from __future__ import annotations

import os
import sys

from decision_harness import DecisionsClient, ModelSelector, ToolSelector
from decision_harness.middleware.model_selection import ModelSelectionMiddleware
from decision_harness.middleware.tool_selection import ToolSelectionMiddleware


def main() -> None:
    try:
        from deepagents import create_deep_agent
        from deepagents.backends.local_shell import LocalShellBackend
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError:
        raise SystemExit("deepagents is not installed: pip install 'decision_harness[deepagents]'")

    args = sys.argv[1:]
    workspace = args[0] if args and os.path.isdir(args[0]) else "."
    task = " ".join(args[1:] if workspace != "." else args) or (
        "Add a subtract(a, b) function to calc.py, cover it in the tests, and run them."
    )

    client = DecisionsClient()
    model_mw = ModelSelectionMiddleware(selector=ModelSelector(client=client))
    tool_mw = ToolSelectionMiddleware(selector=ToolSelector(client=client))

    agent = create_deep_agent(
        model="openai:gpt-6-astra",  # default & escalation ceiling
        middleware=[tool_mw, model_mw],
        system_prompt="You are a helpful coding assistant. Be concise.",
        backend=LocalShellBackend(root_dir=workspace),
        checkpointer=InMemorySaver(),
    )

    print(f"workspace: {os.path.abspath(workspace)}")
    print(f"task:      {task}\n")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"configurable": {"thread_id": "demo"}},
    )
    print(f"reply: {result['messages'][-1].text[:400]}")

    totals = model_mw.ledger.totals()
    print(f"\nmodel calls: {totals['grand']['calls']}, cost: ${totals['grand']['cost']:.6f}")
    for entry in model_mw.ledger.entries[:1]:
        print("decision events:", *entry.events, sep="\n  ")


if __name__ == "__main__":
    main()
