"""Core data types: tiers, model specs, classifications, decisions, selections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Tier(str, Enum):
    """Coarse capability/cost band. Concrete models are attached in the catalog."""

    FAST = "fast"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

    @classmethod
    def order(cls) -> tuple["Tier", ...]:
        return (cls.FAST, cls.BALANCED, cls.PERFORMANCE)

    @classmethod
    def next(cls, tier: "Tier") -> "Tier":
        tiers = cls.order()
        return tiers[min(tiers.index(tier) + 1, len(tiers) - 1)]

    @classmethod
    def strongest(cls, a: "Tier", b: "Tier") -> "Tier":
        return a if cls.order().index(a) >= cls.order().index(b) else b


@dataclass(slots=True)
class ModelSpec:
    """One routable model entry. Everything is data so the catalog is editable."""

    id: str
    provider: str
    tier: Tier = Tier.BALANCED
    # LangChain provider mapping: which init_chat_model provider speaks this
    # entry ("openai", "anthropic", "google_genai"), plus base_url for
    # OpenAI-compatible vendors (fireworks, baseten, openrouter).
    chat_provider: str = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    input_price: float = 0.0  # USD per 1M input tokens (PLACEHOLDER in defaults)
    output_price: float = 0.0  # USD per 1M output tokens (PLACEHOLDER in defaults)
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    reasoning_effort: str | None = None  # pinned reasoning effort for this entry
    capabilities: frozenset[str] = frozenset()  # {"tools","vision","json","reasoning"}
    intelligence: float = 0.0  # relative quality score, 0-10
    aliases: tuple[str, ...] = ()
    notes: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.id)

    def label(self) -> str:
        return f"{self.provider}/{self.id}"

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1e6) * self.input_price + (output_tokens / 1e6) * self.output_price

    def expected_cost(self, est_input_tokens: int = 1_000, est_output_tokens: int = 256) -> float:
        return self.cost(est_input_tokens, est_output_tokens)


@dataclass(slots=True)
class TaskClassification:
    """Typed result of classifying a task (the Decisions answers)."""

    tier: Tier = Tier.BALANCED
    tier_confidence: float = 0.5  # probability of the tier choice
    complexity: float = 1.0  # 0=mechanical .. 3=high consequence
    needs_planning: float = 0.0  # probability the task needs up-front planning
    task_type: str = "other"
    required_capabilities: frozenset[str] = frozenset()
    est_input_tokens: int = 0
    est_output_tokens: int = 512
    classifier: str = ""
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    """Which physical model was chosen, why, and what was considered."""

    model: ModelSpec
    tier: Tier
    reason: str = "classified"  # classified | sticky | escalated | ...
    events: list[str] = field(default_factory=list)
    classification: TaskClassification | None = None
    sticky: bool = False


@dataclass(slots=True)
class ToolSelection:
    """Typed result of tool selection: which tools, how sure, why."""

    selected: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    latency_ms: int = 0


@dataclass(slots=True)
class BreakerConfig:
    """Circuit-breaker settings for the Decisions client."""

    consecutive_failures: int = 3  # failures before opening
    cooldown_s: float = 30.0  # open-state duration before half-open
