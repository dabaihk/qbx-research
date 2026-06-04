# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd

from qbx_research import (
    DeflatedSharpe,
    EvaluatedCandidate,
    Objective,
    PBOResult,
    Strategy,
    StrategyReport,
    basic_metrics,
    evaluate,
)
from qbx_research.demos import MovingAverageCrossover, synthetic_prices


def test_synthetic_prices_are_deterministic():
    a = synthetic_prices(200, seed=7)
    b = synthetic_prices(200, seed=7)
    assert a.equals(b)
    assert len(a) == 200 and (a > 0).all()


def test_demo_strategy_implements_interface():
    strat = MovingAverageCrossover(seed=1)
    assert isinstance(strat, Strategy)
    assert strat.space.names == ["fast", "slow"]
    returns = strat.backtest({"fast": 10, "slow": 50})
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(strat.prices)


def test_evaluate_runs_full_loop():
    report = evaluate(MovingAverageCrossover(seed=3), method="grid", constraints=["fast < slow"])
    assert isinstance(report, StrategyReport)
    assert report.strategy == "moving_average_crossover"
    # every ranked candidate respects the constraint
    assert all(ec.candidate.params["fast"] < ec.candidate.params["slow"] for ec in report.ranking)
    # ranking is best-first by Sharpe
    sharpes = [ec.metrics["sharpe"] for ec in report.ranking]
    assert sharpes == sorted(sharpes, reverse=True)
    assert isinstance(report.best, EvaluatedCandidate)
    assert isinstance(report.deflated_sharpe, DeflatedSharpe)
    assert isinstance(report.pbo, PBOResult)
    assert report.returns.shape[1] == len(report.ranking)


def test_evaluate_honours_custom_objective():
    objective = Objective.minimize("max_drawdown_pct")
    report = evaluate(
        MovingAverageCrossover(seed=5), objective=objective, constraints=["fast < slow"]
    )
    drawdowns = [ec.metrics["max_drawdown_pct"] for ec in report.ranking]
    assert drawdowns == sorted(drawdowns)


def test_basic_metrics_keys():
    metrics = basic_metrics([0.01, -0.02, 0.015, 0.0, 0.03])
    assert set(metrics) == {
        "sharpe", "mean_return", "total_return", "volatility", "max_drawdown_pct"
    }
    assert metrics["max_drawdown_pct"] >= 0.0


def test_custom_strategy_subclass():
    class AlwaysFlat(Strategy):
        name = "flat"

        @property
        def space(self):
            from qbx_research import Param, SearchSpace

            return SearchSpace([Param.values_of("k", [1, 2, 3])])

        def backtest(self, params):
            return pd.Series([0.0] * 300)

    report = evaluate(AlwaysFlat(), pbo_folds=8)
    assert report.strategy == "flat"
    assert report.best.metrics["sharpe"] == 0.0
