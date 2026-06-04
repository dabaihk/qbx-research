# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""Candidate generation over a :class:`~qbx_research.space.SearchSpace`.

Three sampling strategies, one entry point:

* ``grid`` — the full Cartesian product (deterministic, exhaustive).
* ``random`` — uniform random draws, de-duplicated by canonical value.
* ``latin_hypercube`` — stratified space-filling sampling, better coverage per
  sample than naive random for the same budget.

Generation is pure: it produces parameter assignments and never runs a
backtest. Couple it to your own evaluator, then rank the results with
:func:`qbx_research.objective.rank`.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from ._canonical import canonical_json
from .constraints import evaluate as evaluate_constraint
from .space import Param, SearchSpace

__all__ = ["Candidate", "sweep", "SweepMethod"]

SweepMethod = str  # "grid" | "random" | "latin_hypercube"

_DEFAULT_SAMPLES = 64


@dataclass(frozen=True)
class Candidate:
    """One point in the search space.

    ``param_hash`` is a stable digest of ``params`` — identical assignments
    across runs share a hash, which makes a backtest cache trivial to key.
    """

    index: int
    params: dict[str, Any]
    param_hash: str

    @classmethod
    def of(cls, index: int, params: Mapping[str, Any]) -> Candidate:
        return cls(index=index, params=dict(params), param_hash=canonical_json(params))


def sweep(
    space: SearchSpace,
    *,
    method: SweepMethod = "grid",
    budget: int | None = None,
    seed: int = 0,
    constraints: Sequence[str] = (),
) -> list[Candidate]:
    """Generate candidates from ``space``.

    Parameters
    ----------
    space:
        The :class:`SearchSpace` to sample.
    method:
        ``"grid"``, ``"random"`` or ``"latin_hypercube"``.
    budget:
        Maximum number of candidates. For ``grid`` it truncates the product;
        for the sampling methods it sets the sample count (default ``64``).
    seed:
        Seed for the (stochastic) sampling methods. ``grid`` is deterministic
        and ignores it.
    constraints:
        Optional expressions (see :mod:`qbx_research.constraints`) evaluated
        against each candidate's params; only passing candidates are returned.

    Returns
    -------
    list[Candidate]
        Re-indexed contiguously after constraint filtering.
    """
    raw = _generate(space, method=method, budget=budget, seed=seed)
    kept = [params for params in raw if _passes(params, constraints)]
    return [Candidate.of(index, params) for index, params in enumerate(kept)]


def _passes(params: Mapping[str, Any], exprs: Sequence[str]) -> bool:
    for expr in exprs:
        if not expr:
            continue
        if not bool(evaluate_constraint(expr, params)):
            return False
    return True


def _generate(
    space: SearchSpace, *, method: SweepMethod, budget: int | None, seed: int
) -> list[dict[str, Any]]:
    params = list(space)
    if not params:
        return [{}]
    names = [param.name for param in params]

    if method == "grid":
        axes = [param.grid() for param in params]
        if any(not axis for axis in axes):
            raise ValueError("grid sweep requires every param to enumerate >= 1 value")
        points = [dict(zip(names, combo, strict=True)) for combo in product(*axes)]
        return points[:budget] if budget else points

    rng = random.Random(seed)
    count = int(budget or _DEFAULT_SAMPLES)

    if method == "random":
        return _random_sample(params, count, rng)
    if method == "latin_hypercube":
        return _latin_hypercube(params, count, rng)
    raise ValueError(f"unsupported sweep method: {method!r}")


def _random_sample(
    params: Sequence[Param], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Cap attempts so a tiny discrete space can't spin forever chasing duplicates.
    for _ in range(max(count * 10, count)):
        row = {param.name: param.at(rng.random()) for param in params}
        key = canonical_json(row)
        if key in seen:
            continue
        seen.add(key)
        points.append(row)
        if len(points) >= count:
            break
    return points


def _latin_hypercube(
    params: Sequence[Param], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    # Independent per-axis stratified permutation: each axis is split into
    # `count` equal-probability bins, visited in a shuffled order, so the
    # marginal of every parameter is evenly covered.
    if count <= 0:
        return []
    permutations = {
        param.name: rng.sample(range(count), count) for param in params
    }
    points: list[dict[str, Any]] = []
    for i in range(count):
        row = {}
        for param in params:
            stratum = permutations[param.name][i]
            unit = (stratum + rng.random()) / count
            row[param.name] = param.at(unit)
        points.append(row)
    return points
