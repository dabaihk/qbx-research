# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: define a strategy, sweep it, then deflate the winner.

Run it:

    python examples/quickstart.py

The strategy here is a textbook moving-average crossover on a synthetic price
path — deliberately trivial, so the focus stays on the qbx-research workflow
rather than on any real alpha. To research your own strategy, subclass
``qbx_research.Strategy`` and implement ``space`` and ``backtest``; nothing else
changes.
"""
from __future__ import annotations

from qbx_research import Objective, evaluate
from qbx_research.demos import MovingAverageCrossover


def main() -> None:
    strategy = MovingAverageCrossover(seed=20260604)
    print(f"Strategy : {strategy.label()}")
    print(f"Space    : {strategy.space.names}  ({strategy.space.grid_size()} grid points)")

    # One call runs: sweep -> backtest each -> rank -> deflate -> PBO.
    report = evaluate(
        strategy,
        objective=Objective.maximize("sharpe"),
        method="grid",
        constraints=["fast < slow"],
        pbo_folds=8,
    )

    best = report.best
    print(f"\nBest by in-sample Sharpe: {best.candidate.params}  "
          f"(SR={best.metrics['sharpe']:.3f}, "
          f"maxDD={best.metrics['max_drawdown_pct'] * 100:.1f}%)")

    dsr = report.deflated_sharpe
    print("\nDeflated Sharpe Ratio:")
    print(f"  P(skill > 0) after deflation : {dsr.probability:.3f}")
    print(f"  benchmark SR0 (expected max)  : {dsr.sr0:.3f}")
    print(f"  trials counted                : {dsr.n_trials:.2f}")

    pbo = report.pbo
    verdict = "LIKELY OVERFIT" if pbo.pbo > 0.5 else "plausibly robust"
    print(f"\nProbability of Backtest Overfitting: {pbo.pbo:.2f} "
          f"({pbo.n_combinations} CSCV splits)  ->  {verdict}")


if __name__ == "__main__":
    main()
