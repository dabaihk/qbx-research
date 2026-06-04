# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""Declarative configuration for an evaluation run.

A :class:`RunConfig` captures everything :func:`qbx_research.strategy.evaluate`
needs that is *not* the strategy itself — the search method and budget, the
objective, parameter constraints, and the CSCV fold count. It loads from a dict
or a YAML file, so an experiment is reproducible from a checked-in config rather
than a call site.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from . import _yaml
from .objective import Constraint, Objective

__all__ = ["RunConfig"]


@dataclass(frozen=True)
class RunConfig:
    """Everything about an evaluation run except the strategy.

    Attributes
    ----------
    method:
        Sweep method: ``"grid"``, ``"random"`` or ``"latin_hypercube"``.
    budget:
        Candidate cap (sample count for the stochastic methods).
    seed:
        Seed for stochastic sampling.
    constraints:
        Parameter-relation expressions, e.g. ``["fast < slow"]``.
    objective_metric / objective_direction:
        The primary ranking metric and whether to maximise or minimise it.
    metric_constraints:
        Hard gates on backtest metrics, as ``(metric, op, value)`` triples.
    tie_breakers:
        Secondary ranking goals, as ``(metric, direction)`` pairs.
    pbo_folds:
        CSCV fold count for the overfitting estimate.
    """

    method: str = "grid"
    budget: int | None = None
    seed: int = 0
    constraints: tuple[str, ...] = ()
    objective_metric: str = "sharpe"
    objective_direction: str = "maximize"
    metric_constraints: tuple[tuple[str, str, float], ...] = ()
    tie_breakers: tuple[tuple[str, str], ...] = ()
    pbo_folds: int = 8

    def objective(self) -> Objective:
        """Build the :class:`~qbx_research.objective.Objective` this config describes."""
        obj = Objective(
            metric=self.objective_metric,
            direction=self.objective_direction,
            constraints=tuple(Constraint(m, op, v) for m, op, v in self.metric_constraints),
        )
        for metric, direction in self.tie_breakers:
            obj = obj.with_tie_breaker(metric, direction)
        return obj

    # -- (de)serialisation ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown RunConfig keys: {sorted(unknown)}")
        prepared = dict(data)
        if "constraints" in prepared:
            prepared["constraints"] = tuple(prepared["constraints"])
        if "metric_constraints" in prepared:
            prepared["metric_constraints"] = tuple(
                tuple(rule) for rule in prepared["metric_constraints"]
            )
        if "tie_breakers" in prepared:
            prepared["tie_breakers"] = tuple(tuple(tb) for tb in prepared["tie_breakers"])
        return cls(**prepared)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load a run config from a YAML file (requires the ``yaml`` extra)."""
        return cls.from_dict(_yaml.load_file(path))

    @classmethod
    def from_yaml_string(cls, text: str) -> RunConfig:
        return cls.from_dict(_yaml.load_string(text))
