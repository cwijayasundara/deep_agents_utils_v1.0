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
import time

from dotenv import load_dotenv

load_dotenv()  # pick up API keys from the repo-root .env

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
    workspace = args[0] if args and os.path.isdir(args[0]) else "demo_workspace"
    os.makedirs(workspace, exist_ok=True)
    task = " ".join(args[1:] if args and os.path.isdir(args[0]) else args) or (
        "Create calc.py with add(a, b) and subtract(a, b), then create test_calc.py "
        "covering both functions, and run pytest on it. Do not explore first."
    )

    # DEMO_OFFLINE=1 skips the Decisions API entirely (heuristic classifier,
    # keep-all-tools); the agent still runs, just without smart routing.
    offline = os.getenv("DEMO_OFFLINE") == "1"
    client = None if offline else DecisionsClient()
    model_mw = ModelSelectionMiddleware(selector=ModelSelector(client=client))
    tool_mw = ToolSelectionMiddleware(selector=ToolSelector(client=client, offline=offline))

    agent = create_deep_agent(
        model="openai:gpt-6-astra",  # default & escalation ceiling
        middleware=[tool_mw, model_mw],
        system_prompt=(
            f"You are a fast, efficient coding assistant. All file paths are relative to the "
            f"workspace root ({os.path.abspath(workspace)}); never write outside it. Batch work: "
            f"make all independent tool calls in one step, run tests once, then finish."
        ),
        backend=LocalShellBackend(root_dir=workspace, virtual_mode=False),
        checkpointer=InMemorySaver(),
    )

    print(f"mode:      {'offline (heuristic)' if offline else 'decisions API'}")
    print(f"workspace: {os.path.abspath(workspace)}")
    print(f"task:      {task}\n")
    t0 = time.perf_counter()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"configurable": {"thread_id": "demo"}},
    )
    print(f"reply: {result['messages'][-1].text[:400]}")

    totals = model_mw.ledger.totals()
    wall = time.perf_counter() - t0
    print(f"\nmodel calls: {totals['grand']['calls']}, cost: ${totals['grand']['cost']:.6f}")
    print(f"wall time: {wall:.1f}s (model calls: {sum(e.latency_ms for e in model_mw.ledger.entries) / 1000:.1f}s)")
    for entry in model_mw.ledger.entries:
        print(f"  {entry.provider.split('/')[-1]}/{entry.model.split('/')[-1]}: "
              f"{entry.latency_ms / 1000:.1f}s, out={entry.output_tokens} tok ({entry.reason})")
    for entry in model_mw.ledger.entries[:1]:
        print("decision events:", *entry.events, sep="\n  ")


if __name__ == "__main__":
    main()
