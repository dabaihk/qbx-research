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
reason that strategies which look excellent in research underperform in
production [@Bailey2014; @BaileyPBO2017; @HarveyLiu2015].

`qbx-research` is a small, dependency-light Python package that treats the two
halves of honest strategy research as inseparable: (1) **searching** a parameter
space (grid, random, or Latin-hypercube) and ranking candidates against an
explicit objective, and (2) **deflating** the winner's performance for the very
search that produced it. It provides validated, reference-quality
implementations of the Probabilistic Sharpe Ratio (PSR), the Deflated Sharpe
Ratio (DSR), the Probability of Backtest Overfitting (PBO) via Combinatorial
Symmetric Cross-Validation (CSCV), and the supporting corrections for correlated
trials and autocorrelated returns [@Bailey2014; @BaileyPBO2017; @Lo2002;
@LopezdePrado2018]. The selection statistics are pure functions of a return
series and never observe a user's data, signals, or execution model, so they can
be dropped into any research pipeline.

# Statement of need

The methods that correct for selection bias in backtesting — PSR, DSR, and PBO —
are well established [@Bailey2014; @BaileyPBO2017; @HarveyLiuZhu2016;
@LopezdePrado2018], yet practitioners routinely report a single, undeflated
Sharpe ratio. A practical reason is that correct, easy-to-use, low-dependency
implementations are not readily at hand: performance statistics are usually
welded to a particular backtesting engine, data model, or execution simulator,
which makes the selection-aware corrections hard to reuse in isolation and hard
to audit. The consequence is a reproducibility gap — the very statistic most
needed to judge a backtest is the one most often omitted.

The intended users are quantitative researchers, systematic traders, and
students who must report whether a strategy's performance survives a
selection-aware test, and who want implementations they can read, cite, and
trust. `qbx-research` lowers the cost of doing this correctly to a single
function call, with the aim of making "report the deflated statistic, every
time, before capital moves" the default rather than the exception.

# State of the field

Established Python libraries for performance analysis, such as `empyrical`
[@empyrical] and tools built on it, compute the Sharpe ratio, drawdowns, and
related metrics, but do not provide selection-aware corrections (PSR, DSR, or
PBO). Backtesting engines focus on simulation rather than on quantifying
selection bias. The `MlFinLab` library [@mlfinlab] does implement
backtest-overfitting tooling, but as part of a large machine-learning-for-finance
framework with a correspondingly heavier dependency footprint and tighter
coupling to its own abstractions; an R implementation of CSCV PBO also exists on
CRAN. To our knowledge, there is no small, engine-agnostic Python package that
exposes PSR, DSR, PBO, and the effective-trials/effective-sample-size
corrections as independently callable, numerically validated functions with a
minimal dependency surface. `qbx-research` targets exactly that niche: the
statistics can be used on a bare return series, independent of how the backtests
were produced.

# Software design

Three design decisions shape the package. First, **separation of concerns**: the
selection statistics take only a return series (and, for DSR/PBO, a candidate
return matrix), so the statistical core is decoupled from any engine and is
trivial to audit and to embed. A thin `Strategy`/`SignalStrategy` abstraction and
a one-call `evaluate` harness compose search, backtesting, ranking, and deflation
for convenience, but they are optional — a user may bring their own returns.

Second, **a minimal dependency surface**: the only runtime dependencies are NumPy
and pandas. The standard normal quantile function, required by PSR/DSR, uses
Acklam's rational approximation [@Acklam2003] rather than SciPy, which keeps the
package easy to build and to pin in reproducible research environments.

Third, **fail-closed reporting**: when inputs are too thin to support a statistic
(too few observations, degenerate dispersion, an invalid trial count), the
functions raise a structured `NotComputable` signal with a machine-readable
reason instead of returning a misleading number. The implementations follow the
published formulas closely and were validated to within $10^{-9}$ against an
independent production implementation across all functions, so results are
reproducible and checkable.

# Research impact statement

`qbx-research` is released to make selection-aware reporting routine in
quantitative-finance research and teaching. Its credible near-term impact is to
lower the barrier to applying PSR, DSR, and PBO: because the corrections are
engine-agnostic and depend only on NumPy and pandas, they can be added to an
existing study or course with a single function call and without adopting a new
framework. The package is distributed on the Python Package Index and is intended
to support reproducible evaluation of trading strategies and to serve as a
readable, citable reference implementation of the deflated-Sharpe and
backtest-overfitting methods for researchers and students who need to demonstrate
that a backtested result is robust to the search that produced it.

# Mathematics

When the reported strategy is the best of $N$ trials, the honest benchmark is not
zero but the expected maximum Sharpe under the null of no skill,

$$
SR_0 = \overline{SR} + \sigma_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right)
+ \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right],
$$

where $\Phi^{-1}$ is the standard normal quantile, $\sigma_{SR}$ the dispersion of
Sharpe across candidates, and $\gamma \approx 0.5772$ the Euler–Mascheroni
constant. The Deflated Sharpe Ratio is then the Probabilistic Sharpe Ratio
measured against that benchmark, $\widehat{DSR} = \widehat{PSR}(SR_0)$, read as
the probability of genuine skill after accounting for the search
[@Bailey2014].

# AI usage disclosure

In accordance with JOSS policy, the use of generative AI in this submission is
disclosed here. The author used Anthropic's Claude (via the Claude Code
command-line assistant) to assist with software implementation (code generation,
refactoring, and test scaffolding), with documentation, and with drafting the
text of this paper. The author conceived the project, made all architectural and
design decisions, and independently validated all outputs — including the
numerical agreement (to within $10^{-9}$) between the selection-metric
implementations and an independent reference implementation, and the correctness
of the cited references. All AI-assisted code and text were reviewed, edited, and
verified by the author, who takes full responsibility for the content.

# Acknowledgements

The selection-metric methodology follows the work of Bailey and López de Prado on
the Deflated Sharpe Ratio and the Probability of Backtest Overfitting. The author
received no specific grant funding for this work.

# References
