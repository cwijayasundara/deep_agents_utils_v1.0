"""decision_harness: OpenAI Decisions API-backed model & tool selection for agent harnesses.

Core (selectors, catalog, policy, ledger) imports httpx only. The middlewares
are langchain-based and import lazily — install the `deepagents` extra for them.
"""

from .catalog import ModelCatalog
from .decisions import DecisionsClient
from .errors import ConfigError, SelectorUnavailable
from .ledger import CostLedger, UsageEntry
from .policy import EscalationPolicy, TierFloors
from .selector import ModelSelector
from .tools import ToolSelector
from .types import (
    BreakerConfig,
    ModelSpec,
    RouteDecision,
    TaskClassification,
    Tier,
    ToolSelection,
)

__version__ = "0.1.0"

__all__ = [
    "BreakerConfig",
    "ConfigError",
    "CostLedger",
    "DecisionsClient",
    "EscalationPolicy",
    "ModelCatalog",
    "ModelSelector",
    "ModelSpec",
    "RouteDecision",
    "SelectorUnavailable",
    "TaskClassification",
    "Tier",
    "TierFloors",
    "ToolSelection",
    "ToolSelector",
    "UsageEntry",
    "__version__",
]


_MIDDLEWARE_MODULES = {
    "ModelSelectionMiddleware": "decision_harness.middleware.model_selection",
    "ToolSelectionMiddleware": "decision_harness.middleware.tool_selection",
}


def __getattr__(name: str):  # lazy langchain-only exports
    module_path = _MIDDLEWARE_MODULES.get(name)
    if module_path:
        from importlib import import_module

        return getattr(import_module(module_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
