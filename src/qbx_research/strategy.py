# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""The strategy abstraction and a one-call evaluation harness.

A :class:`Strategy` is the *only* thing a user must implement. It exposes two
things: the parameter space it wants searched, and a ``backtest`` that turns one
parameter assignment into a series of per-period returns. How those returns are
produced — what data, what signals, what execution model — is entirely the
strategy's business and is invisible to this library.

:func:`evaluate` then runs the whole honest-research loop over a strategy:
sweep → backtest each candidate → rank → deflate the winner → estimate the
probability of overfitting. The harness only ever sees *returns*; it never
inspects a strategy's internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .objective import Objective, rank
from .search import Candidate, sweep
from .selection import (
    DeflatedSharpe,
    NotComputable,
    PBOResult,
    compute_pbo,
    deflated_sharpe_from_returns,
)
from .space import SearchSpace

__all__ = ["Strategy", "StrategyReport", "EvaluatedCandidate", "evaluate", "basic_metrics"]


class Strategy(ABC):
    """A parameterised trading rule that can be backtested.

    Subclasses implement :attr:`space` (what to search) and :meth:`backtest`
    (how a given parameter set performs). Everything upstream — the data,
    indicators and execution assumptions — lives inside the subclass and is
    never exposed to the search or selection machinery.
    """

    #: Human-readable identifier; defaults to the class name.
    name: str = ""

    @property
    @abstractmethod
    def space(self) -> SearchSpace:
        """The :class:`~qbx_research.space.SearchSpace` to sweep."""

    @abstractmethod
    def backtest(self, params: Mapping[str, Any]) -> pd.Series:
        """Return the per-period return series for one parameter assignment.

        The index is treated as the time axis; values are simple returns (e.g.
        ``0.001`` for +10 bps). The harness derives every metric from this
        series alone.
        """

    def label(self) -> str:
        return self.name or type(self).__name__


@dataclass(frozen=True)
class EvaluatedCandidate:
    """A candidate paired with the metrics of its backtested return series."""

    candidate: Candidate
    metrics: dict[str, float]


@dataclass(frozen=True)
class StrategyReport:
    """The outcome of :func:`evaluate`.

    Attributes
    ----------
    returns:
        A ``periods × candidates`` matrix, one column per candidate keyed by its
        ``param_hash`` — the raw material for the selection statistics.
    ranking:
        Evaluated candidates, best-first under the objective.
    deflated_sharpe / pbo:
        ``None`` when the sweep was too small to support the statistic (e.g. a
        single candidate, or too few observations for the requested folds).
    """

    strategy: str
    ranking: list[EvaluatedCandidate]
    returns: pd.DataFrame
    deflated_sharpe: DeflatedSharpe | None
    pbo: PBOResult | None

    @property
    def best(self) -> EvaluatedCandidate:
        return self.ranking[0]


def basic_metrics(returns: Sequence[float] | pd.Series) -> dict[str, float]:
    """Summary metrics derived purely from a return series.

    Provides the common objective/constraint targets — ``sharpe``,
    ``mean_return``, ``total_return``, ``volatility``, ``max_drawdown_pct`` —
    so callers can rank and gate without a bespoke metrics layer.
    """
    series = (
        pd.to_numeric(pd.Series(returns), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if series.empty:
        return {
            "sharpe": 0.0,
            "mean_return": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
            "max_drawdown_pct": 0.0,
        }
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    mean = float(series.mean())
    equity = (1.0 + series).cumprod()
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "sharpe": round(mean / std, 10) if std > 0 else 0.0,
        "mean_return": round(mean, 12),
        "total_return": round(float(equity.iloc[-1] - 1.0), 12),
        "volatility": round(std, 12),
        "max_drawdown_pct": round(abs(drawdown), 12),
    }


def evaluate(
    strategy: Strategy,
    *,
    objective: Objective | None = None,
    method: str = "grid",
    budget: int | None = None,
    seed: int = 0,
    constraints: Sequence[str] = (),
    pbo_folds: int = 8,
    n_trials: float | None = None,
) -> StrategyReport:
    """Run the full search → rank → deflate loop over ``strategy``.

    Parameters mirror :func:`~qbx_research.search.sweep` for the search phase,
    plus an :class:`~qbx_research.objective.Objective` (default: maximise
    Sharpe) and the CSCV fold count for the overfitting estimate.

    Selection statistics that cannot be computed for the produced matrix are
    returned as ``None`` rather than raising, so a sweep of any size yields a
    usable report.
    """
    objective = objective or Objective.maximize("sharpe")
    candidates = sweep(
        strategy.space, method=method, budget=budget, seed=seed, constraints=constraints
    )
    if not candidates:
        raise ValueError("sweep produced no candidates (check constraints / space)")

    columns: dict[str, np.ndarray] = {}
    evaluated: list[EvaluatedCandidate] = []
    for candidate in candidates:
        series = pd.to_numeric(
            pd.Series(strategy.backtest(candidate.params)), errors="coerce"
        ).fillna(0.0)
        columns[candidate.param_hash] = series.to_numpy(dtype=float)
        evaluated.append(EvaluatedCandidate(candidate, basic_metrics(series)))

    returns = pd.DataFrame(columns)
    ranking = rank(evaluated, objective)
    best_hash = ranking[0].candidate.param_hash

    try:
        dsr: DeflatedSharpe | None = deflated_sharpe_from_returns(
            returns[best_hash], returns, n_trials=n_trials
        )
    except NotComputable:
        dsr = None
    try:
        pbo: PBOResult | None = compute_pbo(returns, metric="sharpe", folds=pbo_folds)
    except NotComputable:
        pbo = None

    return StrategyReport(
        strategy=strategy.label(),
        ranking=ranking,
        returns=returns,
        deflated_sharpe=dsr,
        pbo=pbo,
    )
