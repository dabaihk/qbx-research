# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

from qbx_research import BacktestPolicy, BacktestResult, RunConfig, simulate
from qbx_research.demos import synthetic_prices


def test_policy_defaults_and_validation():
    p = BacktestPolicy()
    assert p.fill == "next_bar" and p.allow_short is True
    assert p.cost_rate_per_turnover == pytest.approx((1.0 + 0.5) / 1e4)
    with pytest.raises(ValueError):
        BacktestPolicy(fill="teleport")
    with pytest.raises(ValueError):
        BacktestPolicy(max_leverage=0)
    with pytest.raises(ValueError):
        BacktestPolicy(fee_bps_per_side=-1)


def test_policy_yaml_round_trip(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "fee_bps_per_side: 2.0\nslippage_bps_per_side: 1.0\nfill: same_bar\nallow_short: true\n"
    )
    p = BacktestPolicy.from_yaml(path)
    assert p.fee_bps_per_side == 2.0 and p.fill == "same_bar" and p.allow_short is True
    # dict round-trips losslessly
    assert BacktestPolicy.from_dict(p.to_dict()) == p


def test_policy_rejects_unknown_keys():
    with pytest.raises(ValueError):
        BacktestPolicy.from_dict({"fee_bps_per_side": 1.0, "bogus": 9})


def test_simulate_long_only_matches_buy_and_hold_without_costs():
    prices = synthetic_prices(300, seed=1)
    policy = BacktestPolicy(fee_bps_per_side=0, slippage_bps_per_side=0, fill="same_bar")
    full_long = pd.Series(1.0, index=prices.index)
    result = simulate(prices, full_long, policy=policy)
    assert isinstance(result, BacktestResult)
    expected = prices.iloc[-1] / prices.iloc[0] - 1.0
    assert result.total_return == pytest.approx(expected, rel=1e-9)


def test_simulate_charges_costs_on_turnover():
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])  # flat market: no PnL, only costs
    position = pd.Series([1.0, 1.0, 0.0, 1.0])         # trades at t1(in), t2(out), t3(in)
    no_cost = BacktestPolicy(fee_bps_per_side=0, slippage_bps_per_side=0)
    with_cost = BacktestPolicy(fee_bps_per_side=10, slippage_bps_per_side=0)
    free = simulate(prices, position, policy=no_cost)
    costed = simulate(prices, position, policy=with_cost)
    assert free.total_return == pytest.approx(0.0, abs=1e-12)
    assert costed.total_return < 0.0          # costs drag a flat market negative
    assert costed.n_trades >= 2


def test_simulate_no_short_clips_negative_positions():
    prices = synthetic_prices(200, seed=2)
    shorts = pd.Series(-1.0, index=prices.index)
    flat = simulate(prices, shorts, policy=BacktestPolicy(allow_short=False))
    # all positions clipped to zero -> no PnL, no trades
    assert flat.total_return == pytest.approx(0.0, abs=1e-12)
    assert flat.n_trades == 0


def test_next_bar_fill_avoids_lookahead():
    # A price that jumps once: same_bar can capture the jump bar, next_bar cannot.
    prices = pd.Series([100.0, 100.0, 110.0, 110.0])
    signal = pd.Series([0.0, 1.0, 0.0, 0.0])  # signal on the bar before the jump
    no_cost = dict(fee_bps_per_side=0, slippage_bps_per_side=0)
    nb = simulate(prices, signal, policy=BacktestPolicy(fill="next_bar", **no_cost))
    sb = simulate(prices, signal, policy=BacktestPolicy(fill="same_bar", **no_cost))
    assert nb.total_return > 0.0           # holds across the +10% bar
    assert sb.total_return == pytest.approx(0.0, abs=1e-12)  # exits before the jump


def test_runconfig_yaml_builds_objective(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "method: random\nseed: 3\nconstraints: ['fast < slow']\n"
        "objective_metric: sharpe\nobjective_direction: maximize\n"
        "metric_constraints: [['max_drawdown_pct', '<=', 0.2]]\n"
        "tie_breakers: [['max_drawdown_pct', 'minimize']]\npbo_folds: 8\n"
    )
    cfg = RunConfig.from_yaml(path)
    assert cfg.method == "random" and cfg.seed == 3
    obj = cfg.objective()
    assert obj.metric == "sharpe" and obj.direction == "maximize"
    assert obj.constraints[0].metric == "max_drawdown_pct"
    assert obj.tie_breakers[0].metric == "max_drawdown_pct"


def test_runconfig_rejects_unknown_keys():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"method": "grid", "bogus": 1})
