# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pandas as pd
import pytest

from qbx_research import (
    DeflatedSharpe,
    NotComputable,
    compute_pbo,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    effective_num_trials,
    effective_sample_size,
    expected_max_sharpe,
    normal_cdf,
    normal_ppf,
    probabilistic_sharpe_ratio,
)


# -- normal distribution ----------------------------------------------------- #
def test_normal_ppf_inverts_cdf():
    for p in (0.01, 0.25, 0.5, 0.84, 0.975):
        assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_normal_cdf_known_values():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.959963985) == pytest.approx(0.975, abs=1e-6)


# -- PSR --------------------------------------------------------------------- #
def test_psr_is_monotonic_in_observed_sharpe():
    lo = probabilistic_sharpe_ratio(0.5, 0.0, 250, 0.0, 3.0)
    hi = probabilistic_sharpe_ratio(1.5, 0.0, 250, 0.0, 3.0)
    assert 0.0 <= lo < hi <= 1.0


def test_psr_grows_with_sample_length():
    short = probabilistic_sharpe_ratio(0.4, 0.0, 50, 0.0, 3.0)
    long = probabilistic_sharpe_ratio(0.4, 0.0, 5000, 0.0, 3.0)
    assert long > short


# -- expected max & DSR ------------------------------------------------------ #
def test_expected_max_sharpe_increases_with_trials():
    one = expected_max_sharpe(1, 0.5)
    many = expected_max_sharpe(1000, 0.5)
    assert one == 0.0  # a single trial has no selection inflation
    assert many > expected_max_sharpe(10, 0.5) > 0.0


def test_dsr_deflates_as_trials_grow():
    # Observed Sharpe sits between the few-trial and many-trial benchmarks, so
    # the deflation is visible rather than saturating PSR at 1.0.
    common = dict(n_observations=120, skew=0.0, kurtosis=3.0, candidate_sr_std=0.5)
    few = deflated_sharpe_ratio(1.0, n_trials=2, **common)
    many = deflated_sharpe_ratio(1.0, n_trials=500, **common)
    assert isinstance(few, DeflatedSharpe)
    assert many.sr0 > few.sr0  # higher bar with more trials
    assert many.probability < few.probability  # deflated harder
    assert few.probability > 0.5 > many.probability  # crosses the coin-flip line


def test_dsr_parametric_fallback_warns():
    result = deflated_sharpe_ratio(1.2, n_observations=500, skew=0.0, kurtosis=3.0, n_trials=10)
    assert result.sr_std_source == "parametric_null_sharpe_se"
    assert "candidate_sr_std_fallback_standard_error" in result.warnings


def test_dsr_rejects_missing_trials():
    with pytest.raises(NotComputable):
        deflated_sharpe_ratio(1.0, 500, 0.0, 3.0, n_trials=None)


# -- effective sample size / trials ------------------------------------------ #
def test_effective_sample_size_bounds():
    rng = np.random.default_rng(0)
    iid = pd.Series(rng.standard_normal(500))
    n_eff = effective_sample_size(iid)
    assert 2.0 <= n_eff <= 500
    # i.i.d. data should retain most of its sample size
    assert n_eff > 300


def test_effective_num_trials_collapses_for_correlated_candidates():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(400)
    identical = pd.DataFrame({f"c{i}": base for i in range(8)})
    independent = pd.DataFrame(rng.standard_normal((400, 8)), columns=[f"c{i}" for i in range(8)])
    assert effective_num_trials(identical) == pytest.approx(1.0, abs=1e-6)
    assert effective_num_trials(independent) > 6.0


# -- PBO --------------------------------------------------------------------- #
def _matrix(rng, n_periods, n_candidates, edge_col=None, edge=0.0):
    data = rng.standard_normal((n_periods, n_candidates)) * 0.01
    if edge_col is not None:
        data[:, edge_col] += edge
    return pd.DataFrame(data, columns=[f"c{i}" for i in range(n_candidates)])


def test_pbo_high_for_pure_noise():
    rng = np.random.default_rng(7)
    frame = _matrix(rng, 240, 20)
    result = compute_pbo(frame, metric="sharpe", folds=8)
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_combinations == math.comb(8, 4)
    # with no real edge, the in-sample winner overfits roughly half the time
    assert result.pbo > 0.3


def test_pbo_low_when_a_candidate_has_real_edge():
    rng = np.random.default_rng(7)
    noise = compute_pbo(_matrix(rng, 240, 20), folds=8).pbo
    rng = np.random.default_rng(7)
    edged = compute_pbo(_matrix(rng, 240, 20, edge_col=0, edge=0.004), folds=8).pbo
    assert edged < noise


def test_pbo_rejects_invalid_folds():
    rng = np.random.default_rng(0)
    with pytest.raises(NotComputable):
        compute_pbo(_matrix(rng, 240, 10), folds=7)  # odd


# -- end to end -------------------------------------------------------------- #
def test_deflated_sharpe_from_returns_end_to_end():
    rng = np.random.default_rng(3)
    candidates = pd.DataFrame(rng.standard_normal((500, 12)) * 0.01)
    best = candidates.mean(axis=0).idxmax()
    result = deflated_sharpe_from_returns(candidates[best], candidates)
    assert isinstance(result, DeflatedSharpe)
    assert 0.0 <= result.probability <= 1.0
    assert result.n_trials >= 1.0
