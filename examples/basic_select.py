"""Model selection only: classify one task and print the routing decision.

    python examples/basic_select.py ["task"]

Offline by default (heuristic fallback); with DECISIONS_API_KEY/OPENAI_API_KEY
set it hits the real OpenAI Decisions API.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()  # pick up API keys from the repo-root .env

from decision_harness import DecisionsClient, ModelCatalog, ModelSelector


def main() -> None:
    task = " ".join(sys.argv[1:]) or (
        "Design a zero-downtime migration with rollback plans."
    )
    selector = ModelSelector(client=DecisionsClient(), catalog=ModelCatalog.load())
    decision = selector.select(task)

    print(f"task:   {task}")
    print(f"tier:   {decision.classification.tier.value} "
          f"(conf {decision.classification.tier_confidence:.2f}, "
          f"complexity {decision.classification.complexity:.2f}, "
          f"planning {decision.classification.needs_planning:.2f})")
    print(f"model:  {decision.model.label()} ({decision.reason})")
    for event in decision.events:
        print(f"event:  {event}")


if __name__ == "__main__":
    main()
