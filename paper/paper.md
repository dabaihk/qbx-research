---
title: 'qbx-research: Parameter sweeps and selection-aware backtest statistics for quantitative research'
tags:
  - Python
  - quantitative finance
  - backtesting
  - overfitting
  - Sharpe ratio
  - deflated Sharpe ratio
  - hyperparameter search
authors:
  - name: Yichong Bai
    orcid: 0009-0007-0835-0135
    affiliation: 1
affiliations:
  - name: Bai Capital
    index: 1
date: 4 June 2026
bibliography: paper.bib
---

# Summary

When a quantitative researcher searches a space of trading strategies and keeps
the best one, the reported performance of that winner is biased upward *by
construction*: it is the maximum of many noisy trials, and the maximum of pure
noise is large. This phenomenon — **backtest overfitting** — is a dominant
reason strategies that look excellent in research underperform in production
[@Bailey2014; @BaileyPBO2017; @HarveyLiu2015].

`qbx-research` is a small, dependency-light Python package that treats the two
halves of honest strategy research as inseparable: (1) **searching** a parameter
space (grid, random, or Latin-hypercube) and ranking candidates against an
explicit objective, and (2) **deflating** the winner's performance for the very
search that produced it. It provides validated, reference-quality
implementations of the Probabilistic Sharpe Ratio (PSR), the Deflated Sharpe
Ratio (DSR), the Probability of Backtest Overfitting (PBO) via Combinatorial
Symmetric Cross-Validation (CSCV), and the supporting corrections for
correlated trials and autocorrelated returns [@Bailey2014; @BaileyPBO2017;
@Lo2002; @LopezdePrado2018].

The library is deliberately unopinionated about *how* a strategy is backtested.
A user implements a single abstract `Strategy` (a parameter space and a
`backtest` that returns a per-period return series), or the position-based
`SignalStrategy`, and hands it to a one-call `evaluate` harness that runs the
entire loop — sweep, backtest each candidate, rank, deflate, and estimate the
probability of overfitting — returning a structured report. The statistical
core never sees a user's data, signals, or execution model; it operates only on
returns. A small, transparent vectorised backtester and YAML-configurable
execution policy are included for convenience but are not required.

Runtime dependencies are limited to NumPy and pandas. The standard normal
quantile function is provided via Acklam's rational approximation
[@Acklam2003], so no SciPy dependency is needed. The selection-metric
implementations were validated to within $10^{-9}$ against an independent
production implementation across all functions.

# Statement of need

The methods that correct for selection bias in backtesting — PSR, DSR, and
PBO — are well established in the literature [@Bailey2014; @BaileyPBO2017;
@LopezdePrado2018], yet practitioners frequently report a single, undeflated
Sharpe ratio because correct, easy-to-use, dependency-light implementations are
not readily at hand. Existing backtesting frameworks tend to couple performance
statistics tightly to a particular engine, data model, or execution simulator,
which makes the selection-aware statistics hard to reuse in isolation and hard
to audit.

`qbx-research` fills this gap with three design choices:

1. **Separation of concerns.** The selection statistics are pure functions of a
   return series and a candidate return matrix. They can be dropped into any
   research pipeline regardless of how the backtests are produced.
2. **Correctness and auditability.** The implementations follow the published
   formulas closely, are unit-tested, and were checked numerically against an
   independent implementation. The expected-maximum-Sharpe benchmark, the
   skew/kurtosis-adjusted PSR, the CSCV PBO, and the eigenvalue-based effective
   number of trials are each exposed as documented, individually callable
   functions.
3. **Minimal footprint.** NumPy and pandas only; no heavyweight or hard-to-build
   dependencies, which matters for reproducible research environments and for
   embedding in larger systems.

The intended users are quantitative researchers, systematic traders, and
students who need to report whether a strategy's performance survives a
selection-aware test, and who want implementations they can read, cite, and
trust. By making these corrections trivial to apply, the package aims to make
"report the deflated statistic, every time, before capital moves" the default
rather than the exception.

# Functionality

- **Search** (`qbx_research.search`): grid, random, and Latin-hypercube
  candidate generation over a declarative `SearchSpace`, with de-duplication,
  inter-parameter constraints, and stable content hashes for caching.
- **Objectives** (`qbx_research.objective`): ranking against a primary metric
  with hard constraints and tie-breakers.
- **Selection metrics** (`qbx_research.selection`):
  `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `compute_pbo` (CSCV),
  `expected_max_sharpe`, `sharpe_standard_error`, `effective_num_trials`
  (eigenvalue participation ratio of the candidate correlation matrix), and
  `effective_sample_size` (autocorrelation-adjusted).
- **Strategy harness** (`qbx_research.strategy`): the `Strategy` /
  `SignalStrategy` abstractions and a one-call `evaluate` that returns a
  `StrategyReport`.
- **Backtest & config** (`qbx_research.backtest`, `qbx_research.config`): a
  clean-room vectorised engine with a configurable, YAML-loadable
  `BacktestPolicy` (fees, slippage, fill timing, shorting, leverage) and a
  `RunConfig` for the evaluation run.

# Acknowledgements

The selection-metric methodology follows the work of Bailey and López de Prado
on the Deflated Sharpe Ratio and the Probability of Backtest Overfitting.

# References
