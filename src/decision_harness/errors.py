"""Error taxonomy."""

from __future__ import annotations


class ConfigError(Exception):
    """Missing key, unknown model, or otherwise invalid configuration."""


class SelectorUnavailable(Exception):
    """The Decisions API could not be reached or refused the question.

    Callers (middlewares, selectors) degrade gracefully instead of raising
    this to the agent.
    """

    def __init__(self, message: str = "", *, retryable: bool = False):
        super().__init__(message or "selector unavailable")
        self.retryable = retryable
