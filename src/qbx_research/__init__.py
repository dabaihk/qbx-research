# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""qbx-research — parameter sweeps and selection-aware backtest statistics.

A small, dependency-light toolkit for the two halves of honest strategy
research:

1. **Search** a parameter space (grid / random / Latin hypercube) and rank the
   results against an explicit objective.
2. **Deflate** the winner's performance for the selection bias that searching
   introduces — Probabilistic and Deflated Sharpe ratios, and the Probability
   of Backtest Overfitting.

Implement a :class:`Strategy` (its parameter space + a ``backtest`` that returns
a series), then hand it to :func:`evaluate` to run the whole loop. The library
never sees your data or signals — only the returns your backtest produces.

Quick start
-----------
>>> from qbx_research import evaluate
>>> from qbx_research.demos import MovingAverageCrossover
>>> report = evaluate(MovingAverageCrossover(), method="grid")
>>> report.best.candidate.params              # doctest: +SKIP
{'fast': 10, 'slow': 50}

The lower-level primitives (:func:`sweep`, :func:`deflated_sharpe_from_returns`,
:func:`compute_pbo`) are also public if you'd rather wire the steps together
yourself.
"""
from __future__ import annotations

from .backtest import BacktestPolicy, BacktestResult, simulate
from .config import RunConfig
from .constraints import ConstraintError
from .constraints import evaluate as eval_constraint
from .objective import Constraint, Objective, rank, satisfies
from .search import Candidate, sweep
from .selection import (
    DeflatedSharpe,
    NotComputable,
    PBOResult,
    ReturnStats,
    compute_pbo,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    effective_num_trials,
    effective_sample_size,
    expected_max_sharpe,
    normal_cdf,
    normal_ppf,
    probabilistic_sharpe_ratio,
    return_stats,
    sharpe_standard_error,
)
from .space import Param, SearchSpace
from .strategy import (
    EvaluatedCandidate,
    SignalStrategy,
    Strategy,
    StrategyReport,
    basic_metrics,
    evaluate,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    # strategy + harness
    "Strategy",
    "SignalStrategy",
    "StrategyReport",
    "EvaluatedCandidate",
    "evaluate",
    "basic_metrics",
    # backtest + config
    "simulate",
    "BacktestPolicy",
    "BacktestResult",
    "RunConfig",
    # search
    "Param",
    "SearchSpace",
    "Candidate",
    "sweep",
    # objective
    "Objective",
    "Constraint",
    "rank",
    "satisfies",
    # constraints
    "eval_constraint",
    "ConstraintError",
    # selection / overfitting
    "ReturnStats",
    "DeflatedSharpe",
    "PBOResult",
    "NotComputable",
    "return_stats",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "deflated_sharpe_from_returns",
    "compute_pbo",
    "expected_max_sharpe",
    "sharpe_standard_error",
    "effective_num_trials",
    "effective_sample_size",
    "normal_cdf",
    "normal_ppf",
]
