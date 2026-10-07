"""ModelSelector: task -> typed classification -> escalation policy -> RouteDecision.

One Decisions request per uncached task. Circuit open or Decisions failure
degrades to the deterministic offline classifier - the selector never breaks
the caller's flow.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import replace
from typing import Callable

from .catalog import ModelCatalog
from .decisions import DecisionsClient, classify_payload
from .heuristic import OfflineFallbackClassifier
from .policy import EscalationPolicy
from .types import TaskClassification, Tier


class ModelSelector:
    def __init__(
        self,
        *,
        client: DecisionsClient,
        catalog: ModelCatalog | None = None,
        policy: EscalationPolicy | None = None,
        cache_size: int = 256,
        on_decision: Callable[[object], None] | None = None,
    ) -> None:
        self.client = client
        self.catalog = catalog if catalog is not None else ModelCatalog.load()
        self.policy = policy if policy is not None else EscalationPolicy()
        self.cache_size = cache_size
        self.on_decision = on_decision
        self._cache: OrderedDict[str, TaskClassification] = OrderedDict()
        self._fallback = OfflineFallbackClassifier()

    # -- sync / async entry points --------------------------------------------

    def select(self, task: str) -> object:
        return self._decide(self._classification(task))

    async def aselect(self, task: str) -> object:
        return self._decide(await self._classification_async(task))

    def _decide(self, classification: TaskClassification) -> object:
        decision = self.policy.decide(classification, self.catalog, reason="user")
        if classification.raw.get("classifier_fallback"):
            decision.events.append(
                f"classifier fallback: decisions unavailable, degraded to heuristic"
            )
        if self.on_decision:
            self.on_decision(decision)
        return decision

    # -- classification (cached, with fallback) --------------------------------

    def _classification(self, task: str) -> TaskClassification:
        cached = self._cache.get(task)
        if cached is not None:
            self._cache.move_to_end(task)
            return replace(cached, classifier="cache")
        try:
            classification = self._classify_via_decisions(task)
        except Exception:  # SelectorUnavailable is the expected shape; degrade regardless
            classification = self._fallback.classify(task)
            classification.raw = {**classification.raw, "classifier_fallback": "decisions"}
        self._remember(task, classification)
        return classification

    async def _classification_async(self, task: str) -> TaskClassification:
        cached = self._cache.get(task)
        if cached is not None:
            self._cache.move_to_end(task)
            return replace(cached, classifier="cache")
        try:
            started = time.perf_counter()
            data = await self.client.apost(classify_payload(task, zero_data_retention=self.client.zero_data_retention))
            flat = self.client._parse_classify(data)
            classification = self._to_classification(flat, started)
        except Exception:
            classification = self._fallback.classify(task)
            classification.raw = {**classification.raw, "classifier_fallback": "decisions"}
        self._remember(task, classification)
        return classification

    # -- helpers ---------------------------------------------------------------

    def _classify_via_decisions(self, task: str) -> TaskClassification:
        started = time.perf_counter()
        data = self.client.post(classify_payload(task, zero_data_retention=self.client.zero_data_retention))
        flat = self.client._parse_classify(data)
        return self._to_classification(flat, started)

    def _to_classification(self, flat: dict, started: float) -> TaskClassification:
        caps_raw = flat.get("required_capabilities")
        classification = TaskClassification(
            tier=Tier(flat["tier"]) if flat["tier"] in Tier._value2member_map_ else Tier.BALANCED,
            tier_confidence=float(flat.get("tier_confidence", 0.5)),
            complexity=float(flat.get("complexity", 1.0)),
            needs_planning=float(flat.get("needs_planning", 0.0)),
            classifier=f"decisions:{self.client.model}",
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw={"answers": flat.get("_answers")},
        )
        classification.est_input_tokens = max(len(str(flat)) // 4, 64)
        if isinstance(caps_raw, list):
            classification.required_capabilities = frozenset(str(c) for c in caps_raw)
        return classification

    def _remember(self, task: str, classification: TaskClassification) -> None:
        if self.cache_size <= 0:
            return
        self._cache[task] = classification
        self._cache.move_to_end(task)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
