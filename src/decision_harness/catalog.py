"""Model catalog: a data-driven registry of routable models across providers.

Defaults ship the three Pareto-frontier tier models from the blog post plus
per-provider alternates (fireworks, openai, anthropic, google, baseten,
openrouter). All prices are PLACEHOLDERS - verify before production, or
overlay your own file: `ModelCatalog.load(path="my_catalog.json")`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigError
from .types import ModelSpec, Tier

DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "default_catalog.json"

_VALID_TIERS = {t.value for t in Tier}


def _spec_from_dict(d: dict[str, Any]) -> ModelSpec:
    missing = [k for k in ("id", "provider") if not d.get(k)]
    if missing:
        raise ConfigError(f"catalog entry missing required fields {missing}: {d}")
    tier = d.get("tier", "balanced")
    if tier not in _VALID_TIERS:
        raise ConfigError(f"unknown tier {tier!r} in catalog entry {d['id']}")
    return ModelSpec(
        id=d["id"],
        provider=d["provider"],
        tier=Tier(tier),
        chat_provider=d.get("chat_provider", "openai"),
        base_url=d.get("base_url"),
        api_key_env=d.get("api_key_env"),
        input_price=float(d.get("input_price", 0.0)),
        output_price=float(d.get("output_price", 0.0)),
        context_window=int(d.get("context_window", 128_000)),
        max_output_tokens=int(d.get("max_output_tokens", 8_192)),
        reasoning_effort=d.get("reasoning_effort"),
        capabilities=frozenset(d.get("capabilities", [])),
        intelligence=float(d.get("intelligence", 0.0)),
        aliases=tuple(d.get("aliases", [])),
        notes=d.get("notes", ""),
    )


class ModelCatalog:
    def __init__(self, specs: Iterable[ModelSpec] | None = None) -> None:
        self._specs: dict[tuple[str, str], ModelSpec] = {}
        for spec in specs or []:
            self.register(spec)

    # -- registration ------------------------------------------------------

    def register(self, spec: ModelSpec, replace: bool = True) -> None:
        if spec.key in self._specs and not replace:
            raise ConfigError(f"model {spec.label()} already registered")
        self._specs[spec.key] = spec

    def add_dict(self, data: dict[str, Any]) -> None:
        for entry in data.get("models", []):
            self.register(_spec_from_dict(entry))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCatalog":
        catalog = cls()
        catalog.add_dict(data)
        return catalog

    @classmethod
    def load(cls, path: str | Path | None = None, merge_defaults: bool = True) -> "ModelCatalog":
        """Defaults merged with an optional user overlay (overlay wins)."""
        catalog = cls()
        if merge_defaults:
            catalog.add_dict(json.loads(DEFAULT_CATALOG_PATH.read_text()))
        if path is not None:
            catalog.add_dict(json.loads(Path(path).read_text()))
        return catalog

    # -- queries -------------------------------------------------------------

    def all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get(self, id: str, provider: str | None = None) -> ModelSpec | None:
        if provider:
            return self._specs.get((provider, id))
        matches = [s for s in self._specs.values() if s.id == id or id in s.aliases]
        return matches[0] if matches else None

    def require(self, id: str, provider: str | None = None) -> ModelSpec:
        spec = self.get(id, provider)
        if spec is None:
            raise ConfigError(f"model {id!r} (provider={provider!r}) not in catalog")
        return spec

    def by_tier(self, tier: Tier) -> list[ModelSpec]:
        """Models in a tier, cheapest (blended 1k-in/256-out) first."""
        specs = [s for s in self.all() if s.tier == tier]
        return sorted(specs, key=lambda s: s.expected_cost(1_000, 256))

    def strongest(self) -> ModelSpec:
        specs = self.all()
        if not specs:
            raise ConfigError("catalog has no models")
        return max(specs, key=lambda s: (s.intelligence, -s.expected_cost(1_000, 256)))

    # -- (de)serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "id": s.id,
                    "provider": s.provider,
                    "tier": s.tier.value,
                    "chat_provider": s.chat_provider,
                    "base_url": s.base_url,
                    "api_key_env": s.api_key_env,
                    "input_price": s.input_price,
                    "output_price": s.output_price,
                    "context_window": s.context_window,
                    "max_output_tokens": s.max_output_tokens,
                    "reasoning_effort": s.reasoning_effort,
                    "capabilities": sorted(s.capabilities),
                    "intelligence": s.intelligence,
                    "aliases": list(s.aliases),
                    "notes": s.notes,
                }
                for s in self.all()
            ]
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
