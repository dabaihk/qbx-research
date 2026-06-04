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

from .backtest import BacktestPolicy, simulate
from .config import RunConfig
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

__all__ = [
    "Strategy",
    "SignalStrategy",
    "StrategyReport",
    "EvaluatedCandidate",
    "evaluate",
    "basic_metrics",
]


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


class SignalStrategy(Strategy):
    """A strategy expressed as target positions over a price series.

    Implement :attr:`space` and :meth:`positions`; the ``backtest`` is provided
    for you by running those positions through :func:`qbx_research.backtest.simulate`
    under :attr:`policy`. This is the common case — define the signal, let the
    configurable :class:`~qbx_research.backtest.BacktestPolicy` own the execution
    assumptions (costs, fill timing, shorting).
    """

    #: Price series the positions trade against. Set this in ``__init__``.
    prices: pd.Series
    #: Execution policy; defaults to ``BacktestPolicy()`` if left unset.
    policy: BacktestPolicy = BacktestPolicy()

    @abstractmethod
    def positions(self, params: Mapping[str, Any]) -> pd.Series:
        """Target position per period for one parameter assignment.

        Signed: ``+1`` fully long, ``-1`` fully short. The policy clips for
        ``allow_short``/``max_leverage`` and applies costs and fill timing.
        """

    def backtest(self, params: Mapping[str, Any]) -> pd.Series:
        return simulate(self.prices, self.positions(params), policy=self.policy).returns


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
    config: RunConfig | None = None,
    objective: Objective | None = None,
    method: str | None = None,
    budget: int | None = None,
    seed: int | None = None,
    constraints: Sequence[str] | None = None,
    pbo_folds: int | None = None,
    n_trials: float | None = None,
) -> StrategyReport:
    """Run the full search → rank → deflate loop over ``strategy``.

    The run is described by a :class:`~qbx_research.config.RunConfig` (loadable
    from YAML); any explicit keyword argument overrides the corresponding config
    field, so ``evaluate(strategy)`` and ``evaluate(strategy, method="random")``
    both work without constructing a config.

    Selection statistics that cannot be computed for the produced matrix are
    returned as ``None`` rather than raising, so a sweep of any size yields a
    usable report.
    """
    config = config or RunConfig()
    method = config.method if method is None else method
    budget = config.budget if budget is None else budget
    seed = config.seed if seed is None else seed
    constraints = config.constraints if constraints is None else constraints
    pbo_folds = config.pbo_folds if pbo_folds is None else pbo_folds
    objective = objective or config.objective()

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
