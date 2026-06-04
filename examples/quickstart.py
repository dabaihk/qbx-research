# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: define a strategy, configure it, sweep, then deflate.

Run it:

    python examples/quickstart.py

The strategy is a textbook moving-average crossover on a synthetic price path —
deliberately trivial, so the focus stays on the qbx-research workflow. Both the
backtest policy (costs, fill, shorting) and the run config (search method,
objective, constraints) are loaded from YAML next to this file. To research your
own strategy, subclass ``qbx_research.SignalStrategy`` and implement ``space``
and ``positions``.
"""
from __future__ import annotations

from pathlib import Path

from qbx_research import BacktestPolicy, RunConfig, evaluate
from qbx_research.demos import MovingAverageCrossover

HERE = Path(__file__).parent


def main() -> None:
    policy = BacktestPolicy.from_yaml(HERE / "policy.yaml")
    config = RunConfig.from_yaml(HERE / "run.yaml")

    strategy = MovingAverageCrossover(seed=20260604, policy=policy)
    print(f"Strategy : {strategy.label()}")
    print(f"Space    : {strategy.space.names}  ({strategy.space.grid_size()} grid points)")
    print(f"Policy   : fees={policy.fee_bps_per_side}bps + slip={policy.slippage_bps_per_side}bps "
          f"/side, fill={policy.fill}, short={policy.allow_short}")
    print(f"Run      : method={config.method}, objective={config.objective_metric} "
          f"({config.objective_direction}), constraints={list(config.constraints)}")

    # One call runs: sweep -> backtest each (under the policy) -> rank -> deflate -> PBO.
    report = evaluate(strategy, config=config)

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
