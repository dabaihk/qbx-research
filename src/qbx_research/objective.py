# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""Objectives: how to compare and rank evaluated candidates.

An :class:`Objective` captures a primary metric and direction, optional hard
constraints, and tie-breakers. :func:`rank` orders a list of evaluated
candidates (anything exposing a ``metrics`` mapping) accordingly, pushing
constraint-violating entries to the back.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Objective", "Constraint", "rank", "satisfies"]

_MAXIMIZE = {"max", "maximize", "desc", "higher"}
_MINIMIZE = {"min", "minimize", "asc", "lower"}

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}
_OP_ALIASES = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "==", "neq": "!="}


def _normalize_direction(direction: Any, default: str = "maximize") -> str:
    text = str(direction or default).lower()
    if text in _MAXIMIZE:
        return "maximize"
    if text in _MINIMIZE:
        return "minimize"
    return default


@dataclass(frozen=True)
class Constraint:
    """A hard threshold a candidate's metric must satisfy to be eligible."""

    metric: str
    op: str
    value: float

    def __post_init__(self) -> None:
        op = _OP_ALIASES.get(self.op, self.op)
        if op not in _OPS:
            raise ValueError(f"unsupported constraint operator: {self.op!r}")
        object.__setattr__(self, "op", op)

    def passes(self, metrics: Mapping[str, Any]) -> bool:
        value = _metric(metrics, self.metric)
        if value is None:
            return False
        return _OPS[self.op](value, self.value)


@dataclass(frozen=True)
class _Goal:
    metric: str
    direction: str = "maximize"


@dataclass(frozen=True)
class Objective:
    """A primary goal, optional constraints, and optional tie-breakers."""

    metric: str
    direction: str = "maximize"
    constraints: tuple[Constraint, ...] = ()
    tie_breakers: tuple[_Goal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _normalize_direction(self.direction))

    @classmethod
    def maximize(cls, metric: str, **kwargs: Any) -> Objective:
        return cls(metric=metric, direction="maximize", **kwargs)

    @classmethod
    def minimize(cls, metric: str, **kwargs: Any) -> Objective:
        return cls(metric=metric, direction="minimize", **kwargs)

    def with_tie_breaker(self, metric: str, direction: str = "maximize") -> Objective:
        goal = _Goal(metric, _normalize_direction(direction))
        return Objective(self.metric, self.direction, self.constraints, self.tie_breakers + (goal,))

    @property
    def _goals(self) -> list[_Goal]:
        return [_Goal(self.metric, self.direction), *self.tie_breakers]


# Common metric aliases so "sharpe" and "net_sharpe" resolve interchangeably.
_ALIASES = {
    "net_sharpe": ("net_sharpe", "sharpe"),
    "sharpe": ("sharpe", "net_sharpe"),
    "cagr": ("cagr", "annualized_return"),
    "annualized_return": ("annualized_return", "cagr"),
    "net_return": ("net_return", "total_return"),
    "total_return": ("total_return", "net_return"),
    "max_drawdown_pct": ("max_drawdown_pct", "max_drawdown"),
}


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    for key in _ALIASES.get(name, (name,)):
        if key in metrics:
            value = metrics[key]
            number = _as_float(value)
            if number is not None:
                return number
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(number) or math.isinf(number)) else number


def satisfies(metrics: Mapping[str, Any], objective: Objective) -> bool:
    """True if ``metrics`` satisfies every hard constraint of ``objective``."""
    return all(constraint.passes(metrics) for constraint in objective.constraints)


def rank(candidates: Iterable[Any], objective: Objective) -> list[Any]:
    """Return ``candidates`` ordered best-first under ``objective``.

    Each candidate must expose a ``metrics`` mapping (attribute or dict key).
    Candidates that violate a constraint or lack the primary metric sort after
    all eligible ones, preserving input order among themselves.
    """
    items = list(candidates)
    eligible = [item for item in items if _eligible(item, objective)]
    rejected = [item for item in items if item not in eligible]
    eligible.sort(key=lambda item: _sort_key(item, objective))
    return eligible + rejected


def _eligible(item: Any, objective: Objective) -> bool:
    metrics = _metrics_of(item)
    if _metric(metrics, objective.metric) is None:
        return False
    return satisfies(metrics, objective)


def _sort_key(item: Any, objective: Objective) -> tuple:
    metrics = _metrics_of(item)
    parts: list[float] = []
    for goal in objective._goals:
        value = _metric(metrics, goal.metric)
        if value is None:
            parts.append(math.inf)
        else:
            parts.append(-value if goal.direction == "maximize" else value)
    return tuple(parts)


def _metrics_of(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        metrics = item.get("metrics", item)
    else:
        metrics = getattr(item, "metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}
