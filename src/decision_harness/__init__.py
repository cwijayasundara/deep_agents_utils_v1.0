"""decision_harness: OpenAI Decisions API-backed model & tool selection for agent harnesses."""

from .catalog import ModelCatalog
from .errors import ConfigError, SelectorUnavailable
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
    "ModelCatalog",
    "ConfigError",
    "ModelSpec",
    "RouteDecision",
    "SelectorUnavailable",
    "TaskClassification",
    "Tier",
    "ToolSelection",
    "__version__",
]
