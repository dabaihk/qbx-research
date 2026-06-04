# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
r"""Selection-aware performance statistics for backtest overfitting.

When you search a space of strategies and keep the best, the winner's Sharpe is
inflated simply because it is the maximum of many noisy trials. This module
implements the standard corrections for that selection bias:

* **PSR** — Probabilistic Sharpe Ratio: the probability the true Sharpe exceeds
  a benchmark, given the sample's length, skew and kurtosis.
* **DSR** — Deflated Sharpe Ratio: PSR against a benchmark set to the *expected
  maximum* Sharpe under the null of no skill across ``n_trials`` correlated
  trials.
* **PBO** — Probability of Backtest Overfitting via Combinatorial Symmetric
  Cross-Validation (CSCV): the fraction of in-sample winners that underperform
  the median out-of-sample.
* **Effective trials / sample size** — deflate naive counts for correlation
  among candidates and autocorrelation within a return series.

References
----------
Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    *Journal of Portfolio Management*, 40(5).
Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017).
    "The Probability of Backtest Overfitting." *Journal of Computational
    Finance*, 20(4).
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "NotComputable",
    "ReturnStats",
    "DeflatedSharpe",
    "PBOResult",
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

_EULER_MASCHERONI = 0.5772156649015329


class NotComputable(ValueError):
    """Raised when a statistic cannot be computed from the given inputs.

    The ``reason`` attribute carries a short machine-readable code (e.g.
    ``"too_few_candidates"``) so pipelines can branch without string-matching
    the message.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# Normal distribution helpers
# --------------------------------------------------------------------------- #
def normal_cdf(x: float) -> float:
    """Standard normal CDF, Φ(x)."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Standard normal inverse CDF, Φ⁻¹(p), via Acklam's rational approximation.

    Accurate to ~1.15e-9 relative error across (0, 1) — ample for Sharpe-scale
    quantiles and dependency-free (no SciPy required).
    """
    p = float(p)
    if p <= 0 or p >= 1:
        raise ValueError("normal_ppf requires probability in the open interval (0, 1)")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# --------------------------------------------------------------------------- #
# Return-series statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReturnStats:
    """Per-period Sharpe and higher moments of a return series."""

    sharpe: float
    n_observations: int
    effective_n: float
    skew: float
    kurtosis: float


def return_stats(returns: Sequence[float] | pd.Series) -> ReturnStats:
    """Summarise a return series for PSR/DSR.

    ``effective_n`` is the autocorrelation-adjusted sample size, used in place
    of the raw count so that serially correlated returns are not credited with
    more independent information than they carry.
    """
    series = (
        pd.to_numeric(pd.Series(returns), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(series) < 2:
        raise NotComputable("insufficient_observations")
    std = float(series.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        raise NotComputable("degenerate_returns")
    mean = float(series.mean())
    pop_std = float(series.std(ddof=0))
    centered = series - mean
    skew = float((centered**3).mean() / pop_std**3) if pop_std > 0 else 0.0
    kurtosis = float((centered**4).mean() / pop_std**4) if pop_std > 0 else 3.0
    return ReturnStats(
        sharpe=round(mean / std, 10),
        n_observations=int(len(series)),
        effective_n=round(effective_sample_size(series), 6),
        skew=round(skew, 10),
        kurtosis=round(kurtosis, 10),
    )


def effective_sample_size(returns: Sequence[float] | pd.Series, max_lag: int = 20) -> float:
    r"""Autocorrelation-adjusted sample size, :math:`n / (1 + 2\sum_k \rho_k^+)`.

    Only positive autocorrelations inflate the variance term, so the result is
    bounded above by ``n`` and below by 2.
    """
    series = (
        pd.to_numeric(pd.Series(returns), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    n = len(series)
    if n < 5:
        return float(n)
    centered = series - float(np.mean(series))
    var = float(np.dot(centered, centered))
    if var <= 0:
        return float(n)
    inflation = 1.0
    for lag in range(1, min(int(max_lag), n - 1) + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / var)
        if math.isfinite(rho):
            inflation += 2.0 * max(0.0, rho)
    return max(2.0, float(n) / inflation)


# --------------------------------------------------------------------------- #
# Sharpe ratio inference
# --------------------------------------------------------------------------- #
def sharpe_standard_error(
    observed_sr: float, n_observations: float, skew: float, kurtosis: float
) -> float:
    r"""Standard error of an estimated Sharpe ratio (Lo, 2002; Mertens, 2002).

    :math:`\mathrm{SE}(\widehat{SR}) =
    \sqrt{(1 - \gamma_3 SR + \tfrac{\gamma_4 - 1}{4} SR^2) / (n - 1)}`,
    where ``γ₃`` is skew and ``γ₄`` is kurtosis.
    """
    radicand = (
        1
        - float(skew) * float(observed_sr)
        + ((float(kurtosis) - 1) / 4) * float(observed_sr) ** 2
    )
    if -1e-12 < radicand < 0:
        radicand = 0.0
    if radicand <= 0 or float(n_observations) < 2:
        return float("nan")
    return math.sqrt(radicand / (float(n_observations) - 1))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: float,
    skew: float,
    kurtosis: float,
) -> float:
    r"""Probabilistic Sharpe Ratio: :math:`\Pr(SR > SR^\*)`.

    The probability that the true Sharpe exceeds ``benchmark_sr``, adjusting the
    estimator's variance for sample length, skew and kurtosis.
    """
    sr = float(observed_sr)
    denom = 1 - float(skew) * sr + ((float(kurtosis) - 1) / 4) * sr**2
    if float(n_observations) < 2 or not math.isfinite(denom):
        raise ValueError("PSR requires finite inputs and at least two observations")
    if -1e-12 < denom < 0:
        denom = 0.0
    if denom <= 0:
        raise ValueError("PSR variance term is not positive")
    z = ((sr - float(benchmark_sr)) * math.sqrt(float(n_observations) - 1)) / math.sqrt(denom)
    return _clamp_probability(normal_cdf(z))


def expected_max_sharpe(n_trials: float, sr_std: float, *, sr_mean: float = 0.0) -> float:
    r"""Expected maximum Sharpe across ``n_trials`` independent null trials.

    The order-statistic benchmark :math:`SR_0` used by the Deflated Sharpe
    Ratio (Bailey & López de Prado, 2014, eq. 6):

    :math:`SR_0 = \bar{SR} + \sigma_{SR}\,[(1-\gamma)\,Z^{-1}(1-1/N)
    + \gamma\,Z^{-1}(1 - 1/(Ne))]`, with ``γ`` the Euler–Mascheroni constant.
    """
    if float(n_trials) <= 1:
        return float(sr_mean)
    n = max(2.0, float(n_trials))
    return float(sr_mean) + float(sr_std) * (
        (1 - _EULER_MASCHERONI) * normal_ppf(1 - 1 / n)
        + _EULER_MASCHERONI * normal_ppf(1 - 1 / (n * math.e))
    )


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeflatedSharpe:
    """Result of a Deflated Sharpe Ratio computation.

    ``probability`` is the deflated probability the strategy has positive true
    skill; values at or above ~0.95 are the usual acceptance bar. ``sr0`` is the
    expected-maximum-Sharpe benchmark the observed Sharpe had to clear.
    """

    probability: float
    sr0: float
    sr_std: float
    sr_std_source: str
    n_trials: float
    formula_version: str = "dsr_v1"
    warnings: tuple[str, ...] = ()


def deflated_sharpe_ratio(
    observed_sr: float,
    n_observations: float,
    skew: float,
    kurtosis: float,
    n_trials: float | None,
    *,
    candidate_sr_std: float | None = None,
    candidate_sr_mean: float | None = None,
    sr0_mode: str = "strict_null",
    allow_parametric_fallback: bool = True,
) -> DeflatedSharpe:
    """Deflate a Sharpe ratio for selection across ``n_trials`` trials.

    The dispersion of Sharpe across candidates (``candidate_sr_std``) is the
    preferred scale for the null. When it is unavailable and
    ``allow_parametric_fallback`` is set, the parametric Sharpe standard error
    is substituted and a warning is recorded.

    Raises
    ------
    NotComputable
        If ``n_trials`` is missing/invalid, observations are too few, or no
        usable Sharpe dispersion can be derived.
    """
    warnings: list[str] = []
    if n_trials is None:
        raise NotComputable("n_trials_missing")
    if float(n_trials) < 1:
        raise NotComputable("n_trials_invalid")
    if float(n_observations) < 2:
        raise NotComputable("insufficient_observations")

    sr_std_source = "candidate_cross_section"
    trial_std = _finite(candidate_sr_std)
    if trial_std is None or trial_std <= 0:
        if not allow_parametric_fallback:
            raise NotComputable("missing_candidate_sharpe_dispersion")
        trial_std = sharpe_standard_error(
            float(observed_sr), float(n_observations), float(skew), float(kurtosis)
        )
        sr_std_source = "parametric_null_sharpe_se"
        warnings.append("candidate_sr_std_fallback_standard_error")
    if trial_std is None or not math.isfinite(float(trial_std)) or float(trial_std) <= 0:
        raise NotComputable("invalid_sharpe_dispersion")

    sr_mean_used = float(candidate_sr_mean or 0.0) if sr0_mode == "empirical_mean" else 0.0
    if sr0_mode != "empirical_mean":
        sr0_mode = "strict_null"
    sr0 = expected_max_sharpe(float(n_trials), float(trial_std), sr_mean=sr_mean_used)
    try:
        probability = probabilistic_sharpe_ratio(
            float(observed_sr), float(sr0), float(n_observations), float(skew), float(kurtosis)
        )
    except ValueError as exc:
        raise NotComputable(str(exc)) from exc
    return DeflatedSharpe(
        probability=round(probability, 10),
        sr0=sr0,
        sr_std=float(trial_std),
        sr_std_source=sr_std_source,
        n_trials=float(n_trials),
        warnings=tuple(warnings),
    )


def deflated_sharpe_from_returns(
    selected_returns: Sequence[float] | pd.Series,
    candidate_returns: pd.DataFrame | None = None,
    *,
    n_trials: float | None = None,
) -> DeflatedSharpe:
    """High-level DSR straight from return series.

    Computes the selected strategy's moments, derives the cross-sectional
    Sharpe dispersion and the effective number of trials from the full
    ``candidate_returns`` matrix (rows = periods, columns = candidates), and
    deflates accordingly. ``n_trials`` overrides the inferred effective count.
    """
    stats = return_stats(selected_returns)
    sr_std = sr_mean = None
    inferred_trials = n_trials
    if candidate_returns is not None and not candidate_returns.empty:
        per_candidate = _candidate_sharpes(candidate_returns)
        if len(per_candidate) > 1:
            sr_std = float(per_candidate.std(ddof=1))
            sr_mean = float(per_candidate.mean())
        if inferred_trials is None:
            inferred_trials = effective_num_trials(candidate_returns)
    if inferred_trials is None:
        # Fall back to the candidate column count, else a single trial.
        inferred_trials = (
            float(candidate_returns.shape[1])
            if candidate_returns is not None and not candidate_returns.empty
            else 1.0
        )
    return deflated_sharpe_ratio(
        stats.sharpe,
        stats.effective_n,
        stats.skew,
        stats.kurtosis,
        inferred_trials,
        candidate_sr_std=sr_std,
        candidate_sr_mean=sr_mean,
    )


# --------------------------------------------------------------------------- #
# Effective number of (correlated) trials
# --------------------------------------------------------------------------- #
def effective_num_trials(candidate_returns: pd.DataFrame) -> float | None:
    r"""Effective number of independent trials from a candidate return matrix.

    Correlated candidates count as fewer independent bets. Using the
    eigenvalues ``λ`` of the candidate correlation matrix:
    :math:`N_\mathrm{eff} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2`
    (a participation-ratio / inverse-Herfindahl measure). Returns ``None`` when
    fewer than two non-degenerate candidates remain.
    """
    if candidate_returns is None or candidate_returns.empty or candidate_returns.shape[1] < 2:
        return None
    frame = (
        candidate_returns.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=1, how="all")
        .dropna(axis=0, how="any")
    )
    std = frame.std(axis=0, ddof=1)
    frame = frame.loc[:, std.replace([np.inf, -np.inf], np.nan).fillna(0.0) > 0.0]
    if frame.shape[1] < 2 or frame.shape[0] < 2:
        return None
    corr = np.corrcoef(frame.to_numpy(dtype=float).T)
    if corr.ndim != 2 or corr.shape[0] < 2:
        return None
    eigvals = np.maximum(np.linalg.eigvalsh(corr), 0.0)
    denom = float(np.sum(eigvals**2))
    if denom <= 0:
        return 1.0
    return float(float(np.sum(eigvals)) ** 2 / denom)


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PBOResult:
    """Result of a CSCV Probability-of-Backtest-Overfitting computation.

    ``pbo`` is the fraction of train/test splits in which the in-sample winner
    landed below the out-of-sample median (logit < 0). Lower is better; the
    common alarm threshold is 0.5, with research pipelines often gating at 0.2.
    """

    pbo: float
    metric: str
    folds: int
    n_combinations: int
    logits: tuple[float, ...]
    oos_percentiles: tuple[float, ...]
    warnings: tuple[str, ...] = ()


def compute_pbo(
    candidate_returns: pd.DataFrame,
    *,
    metric: str = "sharpe",
    folds: int = 8,
) -> PBOResult:
    """Estimate the Probability of Backtest Overfitting via CSCV.

    The period axis (rows) is split into ``folds`` contiguous slices; for every
    way of choosing ``folds/2`` of them as the in-sample set, the in-sample best
    candidate is ranked out-of-sample on the complementary slices. PBO is the
    share of splits whose winner ranks below the OOS median.

    Parameters
    ----------
    candidate_returns:
        Matrix with rows = periods, columns = candidates.
    metric:
        Selection objective; one of ``sharpe``, ``mean_return``,
        ``total_return``, ``volatility``, ``max_drawdown``.
    folds:
        Even integer >= 4. Yields ``C(folds, folds/2)`` evaluations.

    Raises
    ------
    NotComputable
        If the matrix is too small or ``folds`` is invalid.
    """
    metric = _resolve_pbo_metric(metric)
    if candidate_returns is None or candidate_returns.empty:
        raise NotComputable("candidate_returns_missing")
    frame = (
        candidate_returns.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=1, how="all")
        .dropna(axis=0, how="all")
        .fillna(0.0)
    )
    if frame.shape[1] < 2:
        raise NotComputable("too_few_candidates")
    if int(folds) < 4 or int(folds) % 2 != 0:
        raise NotComputable("invalid_folds")
    if frame.shape[0] < max(4, int(folds)):
        raise NotComputable("too_few_observations")
    if int(folds) > frame.shape[0]:
        raise NotComputable("folds_exceed_observations")

    slices = np.array_split(np.arange(frame.shape[0]), int(folds))
    combinations = list(itertools.combinations(range(int(folds)), int(folds) // 2))
    logits: list[float] = []
    percentiles: list[float] = []
    warnings: list[str] = []
    overfit = 0
    for combo in combinations:
        is_rows = np.concatenate([slices[i] for i in combo])
        oos_rows = np.concatenate([slices[i] for i in range(int(folds)) if i not in combo])
        is_scores = _score(frame.iloc[is_rows], metric)
        oos_scores = _score(frame.iloc[oos_rows], metric)
        if is_scores.empty or oos_scores.empty:
            warnings.append("empty_fold_scores")
            continue
        winner = str(is_scores.sort_values(ascending=False).index[0])
        _, percentile, logit = _oos_rank(oos_scores, winner)
        if winner not in oos_scores.index.astype(str):
            warnings.append("winner_missing_oos_score")
        logits.append(round(logit, 10))
        percentiles.append(round(percentile, 10))
        if logit < 0:
            overfit += 1
    if not logits:
        raise NotComputable("no_valid_cscv_combinations")
    return PBOResult(
        pbo=round(overfit / len(logits), 8),
        metric=metric,
        folds=int(folds),
        n_combinations=len(logits),
        logits=tuple(logits),
        oos_percentiles=tuple(percentiles),
        warnings=tuple(dict.fromkeys(warnings)),  # de-dup, order-preserving
    )


# --------------------------------------------------------------------------- #
# Internal scoring helpers
# --------------------------------------------------------------------------- #
_PBO_ALIASES = {
    "net_sharpe": "sharpe",
    "sharpe_ratio": "sharpe",
    "avg_return": "mean_return",
    "average_return": "mean_return",
    "return": "mean_return",
    "net_return": "total_return",
    "drawdown": "max_drawdown",
    "max_drawdown_pct": "max_drawdown",
    "std": "volatility",
}
_PBO_SUPPORTED = {"sharpe", "mean_return", "total_return", "volatility", "max_drawdown"}


def _resolve_pbo_metric(metric: str) -> str:
    name = _PBO_ALIASES.get(str(metric).lower(), str(metric).lower())
    if name not in _PBO_SUPPORTED:
        raise NotComputable(f"unsupported_pbo_objective:{name}")
    return name


def _score(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "sharpe":
        std = frame.std(axis=0, ddof=1).replace(0, np.nan)
        scores = frame.mean(axis=0) / std
    elif metric == "mean_return":
        scores = frame.mean(axis=0)
    elif metric == "total_return":
        scores = (1.0 + frame).prod(axis=0) - 1.0
    elif metric == "volatility":
        scores = -frame.std(axis=0, ddof=1)
    elif metric == "max_drawdown":
        scores = -frame.apply(_max_drawdown)
    else:  # pragma: no cover - guarded by _resolve_pbo_metric
        raise NotComputable(f"unsupported_pbo_objective:{metric}")
    return scores.replace([np.inf, -np.inf], np.nan).dropna()


def _max_drawdown(returns: pd.Series) -> float:
    series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return float("nan")
    equity = (1.0 + series).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _oos_rank(scores: pd.Series, winner: str) -> tuple[float, float, float]:
    clean = scores.dropna()
    labels = [str(item) for item in clean.index]
    if winner not in labels:
        rank = 1.0
        omega = rank / (len(clean) + 1.0)
        return rank, omega, math.log(omega / (1.0 - omega))
    ranks = clean.rank(method="average", ascending=True)
    rank = float(ranks.loc[clean.index[labels.index(winner)]])
    omega = rank / (len(clean) + 1.0)
    return rank, omega, math.log(omega / (1.0 - omega))


def _candidate_sharpes(candidate_returns: pd.DataFrame) -> pd.Series:
    return _score(
        candidate_returns.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0),
        "sharpe",
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
