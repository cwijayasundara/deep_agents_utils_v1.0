"""EscalationPolicy: map a classification to a concrete model decision.

Conservative overrides (the blog's "when in doubt, escalate") - with one
live-validated exception: uncertainty on clearly *mechanical* work stays in
the classified tier, because escalating a mechanical task to a frontier model
is pure waste (A/B feedback: "pretty expensive for this query").
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import ModelCatalog
from .types import RouteDecision, TaskClassification, Tier


@dataclass(slots=True)
class TierFloors:
    route_confidence_floor: float = 0.68  # below -> escalate (blog default)
    complexity_floor: float = 1.75  # at/above -> escalate
    planning_floor: float = 0.6  # at/above -> escalate


class EscalationPolicy:
    def __init__(
        self,
        floors: TierFloors | None = None,
        override_target: Tier = Tier.PERFORMANCE,
        context_margin_tokens: int = 4_000,
    ) -> None:
        self.floors = floors or TierFloors()
        self.override_target = override_target
        self.context_margin_tokens = context_margin_tokens

    def __repr__(self) -> str:  # pragma: no cover
        return f"EscalationPolicy(floors={self.floors!r}, override_target={self.override_target.value})"

    def _pick_in_tier(self, tier: Tier, classification: TaskClassification, catalog: ModelCatalog):
        needed_ctx = (
            classification.est_input_tokens
            + classification.est_output_tokens
            + self.context_margin_tokens
        )
        for spec in catalog.by_tier(tier):
            if (
                classification.required_capabilities <= spec.capabilities
                and spec.context_window >= needed_ctx
            ):
                return spec
        return None

    def decide(
        self,
        classification: TaskClassification,
        catalog: ModelCatalog,
        *,
        reason: str = "user",
    ) -> RouteDecision:
        events: list[str] = []
        f = self.floors
        tier = classification.tier
        overridden = False

        if reason == "retry":
            tier = Tier.next(tier)
            overridden = True
            events.append(f"retry: escalated tier to {tier.value}")

        mechanical = (
            classification.complexity < 1.0
            and classification.needs_planning < f.planning_floor
        )
        if classification.tier_confidence < f.route_confidence_floor:
            if mechanical:
                events.append(
                    f"uncertain route ({classification.tier_confidence:.2f} < "
                    f"{f.route_confidence_floor:.2f}) but mechanical task "
                    f"(complexity {classification.complexity:.2f}) stays in {tier.value}"
                )
            else:
                overridden = True
                tier = Tier.strongest(tier, self.override_target)
                events.append(
                    f"uncertain route ({classification.tier_confidence:.2f} < "
                    f"{f.route_confidence_floor:.2f}) -> {self.override_target.value}"
                )
        if classification.complexity >= f.complexity_floor:
            overridden = True
            tier = Tier.strongest(tier, self.override_target)
            events.append(
                f"complexity {classification.complexity:.2f} >= {f.complexity_floor} "
                f"-> {self.override_target.value}"
            )
        if classification.needs_planning >= f.planning_floor:
            overridden = True
            tier = Tier.strongest(tier, self.override_target)
            events.append(
                f"planning probability {classification.needs_planning:.2f} >= "
                f"{f.planning_floor} -> {self.override_target.value}"
            )

        spec = self._pick_in_tier(tier, classification, catalog)
        if spec is None:
            overridden = True
            events.append(f"no eligible {tier.value} model (caps/context); escalating tier")
            for t in Tier.order()[Tier.order().index(tier) + 1 :]:
                spec = self._pick_in_tier(t, classification, catalog)
                if spec:
                    tier = t
                    break
        if spec is None:
            spec = catalog.strongest()
            events.append(f"fallback: strongest model {spec.label()}")

        return RouteDecision(
            model=spec,
            tier=spec.tier,
            reason="escalated" if overridden else "classified",
            events=events,
            classification=classification,
        )
