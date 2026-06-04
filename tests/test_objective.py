# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
from qbx_research import Constraint, Objective, eval_constraint, rank, satisfies


def test_rank_maximizes_primary_metric():
    items = [
        {"id": "a", "metrics": {"sharpe": 1.0}},
        {"id": "b", "metrics": {"sharpe": 2.5}},
        {"id": "c", "metrics": {"sharpe": 1.7}},
    ]
    ordered = rank(items, Objective.maximize("sharpe"))
    assert [item["id"] for item in ordered] == ["b", "c", "a"]


def test_constraints_push_violators_to_back():
    items = [
        {"id": "good", "metrics": {"sharpe": 2.0, "max_drawdown_pct": 0.1}},
        {"id": "risky", "metrics": {"sharpe": 3.0, "max_drawdown_pct": 0.5}},
    ]
    objective = Objective.maximize(
        "sharpe", constraints=(Constraint("max_drawdown_pct", "<=", 0.2),)
    )
    ordered = rank(items, objective)
    assert [item["id"] for item in ordered] == ["good", "risky"]
    assert satisfies(items[0]["metrics"], objective)
    assert not satisfies(items[1]["metrics"], objective)


def test_tie_breaker_resolves_equal_primary():
    items = [
        {"id": "a", "metrics": {"sharpe": 2.0, "turnover": 0.9}},
        {"id": "b", "metrics": {"sharpe": 2.0, "turnover": 0.2}},
    ]
    objective = Objective.maximize("sharpe").with_tie_breaker("turnover", "minimize")
    assert [item["id"] for item in rank(items, objective)] == ["b", "a"]


def test_metric_aliases_resolve():
    objective = Objective.maximize("sharpe")
    ordered = rank([{"id": "x", "metrics": {"net_sharpe": 1.5}}], objective)
    assert ordered[0]["id"] == "x"  # net_sharpe resolved as sharpe


def test_safe_constraint_evaluation():
    assert eval_constraint("a < b and c <= 0.5", {"a": 1, "b": 2, "c": 0.3}) is True
    assert eval_constraint(
        "fast.window < slow.window", {"fast.window": 5, "slow.window": 20}
    ) is True
