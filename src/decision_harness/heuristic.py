"""Offline fallback classifier: deterministic keyword scoring, zero cost.

Used when the Decisions API is unreachable (circuit open, timeout, refusal).
Never as good as the decision model - but it keeps the harness routing
instead of erroring.
"""

from __future__ import annotations

import re
import time

from .types import TaskClassification, Tier

# Signal words per tier (scored; ties break toward the cheaper tier).
_FAST_SIGNALS = (
    "typo", "rename", "format", "whitespace", "spelling", "grammar",
    "comment", "log message", "readme", "docstring", "import order",
    "aria-label", "alt text", "copy change", "version bump",
)
_BALANCED_SIGNALS = (
    "implement", "add", "feature", "bug", "fix", "endpoint", "function",
    "refactor", "unit test", "validation", "pagination", "caching",
    "upgrade", "migrate", "error handling",
)
_PERFORMANCE_SIGNALS = (
    "architecture", "design", "security", "migration", "zero-downtime",
    "rollback", "cross-system", "scalab", "concurrency", "race condition",
    "diagnose", "investigate", "root cause", "production incident",
    "database schema", "protocol", "threat model", "audit",
)


class OfflineFallbackClassifier:
    """Keyword-scored classification. Deterministic and instant."""

    name = "heuristic"

    def classify(self, task: str) -> TaskClassification:
        started = time.perf_counter()
        text = task.lower()

        fast = sum(1 for s in _FAST_SIGNALS if s in text)
        balanced = sum(1 for s in _BALANCED_SIGNALS if s in text)
        performance = sum(1 for s in _PERFORMANCE_SIGNALS if s in text)

        if performance > 0 and performance >= max(fast, balanced):
            tier, confidence = Tier.PERFORMANCE, 0.75
        elif fast >= balanced:  # ties break toward the cheaper tier
            tier, confidence = Tier.FAST, 0.75
        elif balanced > 0:
            tier, confidence = Tier.BALANCED, 0.65
        else:
            tier, confidence = Tier.FAST, 0.55

        complexity = min(3.0, performance * 1.5 + balanced * 0.4)
        needs_planning = min(1.0, performance * 0.4 + (0.2 if balanced else 0.0))

        return TaskClassification(
            tier=tier,
            tier_confidence=confidence,
            complexity=round(complexity, 2),
            needs_planning=round(needs_planning, 2),
            task_type=self._task_type(text),
            classifier=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw={"fast_signals": fast, "balanced_signals": balanced, "performance_signals": performance},
        )

    @staticmethod
    def _task_type(text: str) -> str:
        if re.search(r"\b(test|spec)\b", text):
            return "test_change"
        if "question" in text or text.endswith("?"):
            return "question"
        if "design" in text or "architecture" in text:
            return "design"
        return "code_change"
