# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
import pytest

from qbx_research import Param, SearchSpace, sweep


def test_grid_is_full_cartesian_product():
    space = SearchSpace([
        Param.values_of("fast", [5, 10]),
        Param.values_of("slow", [20, 30, 40]),
    ])
    candidates = sweep(space, method="grid")
    assert len(candidates) == 6 == space.grid_size()
    assert {(c.params["fast"], c.params["slow"]) for c in candidates} == {
        (5, 20), (5, 30), (5, 40), (10, 20), (10, 30), (10, 40)
    }


def test_grid_respects_budget_truncation():
    space = SearchSpace([Param.values_of("x", list(range(100)))])
    assert len(sweep(space, method="grid", budget=7)) == 7


def test_range_param_int_grid_is_inclusive():
    values = Param.range("p", 2, 6, step=2, dtype="int").grid()
    assert values == [2, 4, 6]


def test_random_is_seed_reproducible_and_deduped():
    space = SearchSpace([Param.range("a", 0.0, 1.0), Param.range("b", 0.0, 1.0)])
    first = [c.params for c in sweep(space, method="random", budget=16, seed=42)]
    second = [c.params for c in sweep(space, method="random", budget=16, seed=42)]
    assert first == second
    keys = {(round(p["a"], 10), round(p["b"], 10)) for p in first}
    assert len(keys) == len(first)  # no duplicates


def test_latin_hypercube_covers_each_stratum_once():
    space = SearchSpace([Param.range("a", 0.0, 1.0)])
    n = 10
    candidates = sweep(space, method="latin_hypercube", budget=n, seed=1)
    assert len(candidates) == n
    strata = {min(int(c.params["a"] * n), n - 1) for c in candidates}
    assert strata == set(range(n))  # every stratum hit exactly once


def test_constraints_filter_and_reindex():
    space = SearchSpace([
        Param.values_of("fast", [5, 10, 20]),
        Param.values_of("slow", [10, 20]),
    ])
    candidates = sweep(space, method="grid", constraints=["fast < slow"])
    assert all(c.params["fast"] < c.params["slow"] for c in candidates)
    assert [c.index for c in candidates] == list(range(len(candidates)))


def test_param_hash_is_order_independent():
    a = sweep(SearchSpace([Param.values_of("x", [1]), Param.values_of("y", [2])]), method="grid")
    assert a[0].param_hash == sweep(
        SearchSpace([Param.values_of("y", [2]), Param.values_of("x", [1])]), method="grid"
    )[0].param_hash


def test_log_scale_requires_positive_low():
    with pytest.raises(ValueError):
        Param.range("p", 0.0, 1.0, scale="log")


def test_param_rejects_both_values_and_range():
    with pytest.raises(ValueError):
        Param(name="p", values=(1, 2), low=0, high=1)
