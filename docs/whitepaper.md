---
title: "Selection-Aware Backtest Statistics: A Dependency-Light Reference Implementation of the Deflated Sharpe Ratio and the Probability of Backtest Overfitting"
author: "Yichong Bai"
date: "June 2026"
abstract: |
  Searching across many candidate strategies and reporting the performance of
  the best one yields a statistic that is biased upward by construction: it is
  the maximum of many noisy trials. We review the standard corrections for this
  selection bias — the Probabilistic and Deflated Sharpe Ratios and the
  Probability of Backtest Overfitting via Combinatorial Symmetric
  Cross-Validation — and present `qbx-research`, a small, open-source Python
  implementation that is dependency-light (NumPy and pandas only), validated to
  within 1e-9 against an independent production implementation, and decoupled
  from any particular backtesting engine. We give the formulas, the software
  design, a fully reproducible worked example, and practical guidance for
  applying these statistics in research.
---

# 1. Introduction

Every backtest is a hypothesis test. Running one and observing a Sharpe ratio of
1.5 carries information. Running a *thousand* — sweeping windows, thresholds,
holding periods, and universes — and reporting the best of them carries far
less, because the reported number is the **maximum** of many noisy estimates,
and the maximum of pure noise is large. This is **backtest overfitting**, and it
is among the most common reasons that strategies which look excellent in
research disappoint in production.

The corrections for this bias are well established. Bailey and López de Prado
(2014) introduced the Deflated Sharpe Ratio (DSR); Bailey, Borwein, López de
Prado, and Zhu (2017) introduced the Probability of Backtest Overfitting (PBO)
via Combinatorial Symmetric Cross-Validation (CSCV); and Lo (2002) gave the
non-normality-adjusted standard error of the Sharpe ratio on which the
probabilistic versions rest. Despite this, practitioners routinely report a
single, undeflated Sharpe ratio. Part of the reason is practical: correct,
easy-to-use, low-dependency implementations are not always at hand, and existing
frameworks tend to weld the statistics to a specific engine and data model.

This paper does two things. First (Sections 2–3) it states the mechanism and the
formulas precisely, in the exact form implemented by the accompanying software.
Second (Sections 4–6) it describes `qbx-research`, an open-source package that
provides these statistics as pure, auditable functions, and demonstrates them on
a reproducible example.

# 2. The selection-bias mechanism

Let a researcher evaluate $N$ candidate strategies. Suppose, as a null, that all
$N$ have zero true skill, so each estimated Sharpe ratio is mean-zero noise with
cross-sectional standard deviation $\sigma_{SR}$. The expected value of the
*best* observed Sharpe is then not zero; using the order statistics of the
normal distribution it is approximately

$$
\mathbb{E}\!\left[\max_{i\le N}\widehat{SR}_i\right] \approx
\sigma_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right)
+\gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right],
$$

where $\Phi^{-1}$ is the standard normal quantile function and
$\gamma\approx0.5772$ is the Euler–Mascheroni constant. This benchmark grows
without bound in $N$: the harder one searches, the higher the bar that a genuine
edge must clear. The correct question is therefore never "what was the best
Sharpe?" but "is the best Sharpe larger than what this search would have
produced from noise alone?"

# 3. Methods

Throughout, a strategy produces $n$ per-period returns with sample mean $\mu$ and
standard deviation $\sigma$; $\hat\gamma_3$ and $\hat\gamma_4$ denote sample
skewness and kurtosis; $\Phi$ is the standard normal CDF.

## 3.1 The Sharpe ratio and its standard error

$$
\widehat{SR}=\frac{\mu}{\sigma},\qquad
\sigma_{\widehat{SR}}=
\sqrt{\frac{1-\hat\gamma_3\,\widehat{SR}+\frac{\hat\gamma_4-1}{4}\,\widehat{SR}^2}{n-1}}.
$$

Negative skew and heavy tails inflate this standard error — precisely the return
profile of strategies that sell insurance (Lo, 2002).

## 3.2 Probabilistic Sharpe Ratio (PSR)

The probability that the true Sharpe exceeds a benchmark $SR^\ast$:

$$
\widehat{PSR}(SR^\ast)=\Phi\!\left(
\frac{(\widehat{SR}-SR^\ast)\sqrt{n-1}}
{\sqrt{1-\hat\gamma_3\,\widehat{SR}+\frac{\hat\gamma_4-1}{4}\,\widehat{SR}^2}}
\right).
$$

## 3.3 The selection benchmark

When the reported strategy is the winner of $N$ trials, the honest benchmark is
the expected maximum Sharpe under the null,

$$
SR_0=\overline{SR}+\sigma_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right)
+\gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right],
$$

with $\sigma_{SR}$ the dispersion of Sharpe across the candidates and
$\overline{SR}$ their mean (taken as $0$ under the strict null).

## 3.4 Deflated Sharpe Ratio (DSR)

The DSR is the PSR measured against the selection-aware benchmark:

$$
\widehat{DSR}=\widehat{PSR}(SR_0).
$$

It is read as the probability that the strategy has genuine skill *after*
accounting for the fact that it was selected. A DSR of $0.95$ is the customary
acceptance threshold.

## 3.5 Probability of Backtest Overfitting (PBO)

PBO interrogates the selection *process* rather than a single number. CSCV
partitions the timeline into $S$ contiguous slices; for each of the
$\binom{S}{S/2}$ ways of choosing half the slices as in-sample, the in-sample
best candidate is ranked out-of-sample. With $\omega_c\in(0,1)$ the relative OOS
rank of the in-sample winner in split $c$,

$$
\lambda_c=\ln\frac{\omega_c}{1-\omega_c},\qquad
\mathrm{PBO}=\frac{1}{|C|}\sum_{c\in C}\mathbf{1}[\lambda_c<0].
$$

A winner that overfits lands below the OOS median ($\lambda_c<0$). $\mathrm{PBO}
\approx 0.5$ means the selection has no out-of-sample value; low PBO means the
winner tends to generalise.

## 3.6 Deflating the trial count

Two corrections keep the inputs honest. Correlated candidates are not
independent bets; from the eigenvalues $\lambda_i$ of the candidate correlation
matrix the effective number of trials is the participation ratio

$$
N_\text{eff}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
$$

Autocorrelated returns carry less information than their length suggests; with
autocorrelations $\rho_k$,

$$
n_\text{eff}=\frac{n}{1+2\sum_{k=1}^{K}\max(0,\rho_k)}.
$$

# 4. Software design

`qbx-research` is organised around three principles.

**Separation of concerns.** The selection statistics are pure functions of a
return series and a candidate return matrix. They are independent of how
backtests are produced, so they slot into any pipeline. A strategy is expressed
by implementing an abstract `Strategy` (a parameter space and a `backtest`
returning a return series) or the position-based `SignalStrategy`; the
`evaluate` harness runs sweep → backtest → rank → deflate → PBO and returns a
structured report. The statistical core never observes a user's data or signals.

**Correctness and auditability.** The expected-maximum-Sharpe benchmark, the
skew/kurtosis-adjusted PSR, the CSCV PBO, and the eigenvalue-based effective
trial count are each individually callable and unit-tested. All selection
functions were validated to within $10^{-9}$ against an independent production
implementation.

**Minimal footprint.** The only runtime dependencies are NumPy and pandas. The
standard normal quantile function uses Acklam's rational approximation, so SciPy
is not required — which matters for reproducible environments and embedding.

The package also ships a small, transparent vectorised backtester and a
YAML-configurable `BacktestPolicy` (fees, slippage, fill timing, shorting,
leverage) and `RunConfig`, but these are conveniences: a user may bring their
own returns.

# 5. Worked example

The following uses the package's bundled demo — a moving-average crossover on a
synthetic geometric-Brownian-motion price path — to make the deflation concrete.

```python
from qbx_research import evaluate, Objective
from qbx_research.demos import MovingAverageCrossover

report = evaluate(
    MovingAverageCrossover(seed=20260604),
    objective=Objective.maximize("sharpe"),
    constraints=["fast < slow"],
    pbo_folds=8,
)
print(report.best.candidate.params, report.best.metrics["sharpe"])
print(report.deflated_sharpe.probability, report.deflated_sharpe.sr0)
print(report.pbo.pbo)
```

A representative run reports an in-sample winner with a modest Sharpe near
$0.07$, a Deflated Sharpe probability of roughly $0.88$ against an
expected-maximum benchmark $SR_0\approx0.01$, and a PBO near $0.21$ over the
$\binom{8}{4}=70$ CSCV splits — a result that is "plausibly robust" for this toy
problem rather than a confident edge. The point of the example is mechanical,
not financial: it shows the entire honest-research loop — search, then deflate
the optimism that searching created — in a handful of lines and with numbers a
reader can reproduce.

# 6. Practical guidance

- **Always report a deflated statistic.** A bare Sharpe selected from a search
  is a measurement error, not a judgement call.
- **Count trials honestly.** Use $N_\text{eff}$, not the raw grid size; a
  redundant sweep of near-identical parameters is a handful of real trials, not
  hundreds.
- **Use $n_\text{eff}$ for serially correlated returns**, especially at high
  frequency, so the sample is not credited with more independent information than
  it carries.
- **Read DSR and PBO together.** DSR corrects a number; PBO resamples the
  decision. A finding should clear both. Customary bars: $\text{DSR}\ge0.95$,
  $\text{PBO}\le0.10$ (and certainly well below $0.5$).
- These thresholds are conventions; calibrate to your own acceptance budget.
  The discipline that matters is reporting them *at all*, before capital moves.

# 7. Limitations

The corrections rest on assumptions worth stating. The expected-maximum-Sharpe
benchmark assumes a (possibly correlated) normal model for trial Sharpe ratios;
heavily non-normal trial distributions weaken it. PSR/DSR adjust for the first
four moments only. CSCV assumes the timeline can be partitioned into
exchangeable blocks; strong regime structure violates this and the fold count
must be chosen sensibly. The effective-trials estimate is a second-moment
(correlation) summary and does not capture higher-order dependence among
candidates. None of these are unique to this implementation; they are properties
of the methods, and the software is explicit about them, raising a structured
"not computable" signal rather than returning a misleading number when inputs
are too thin.

# 8. Availability

`qbx-research` is open source under the Apache License 2.0. It is available on
PyPI (`pip install qbx-research`) and on GitHub at
<https://github.com/dabaihk/qbx-research>. It targets Python 3.10+ and depends
only on NumPy and pandas (with PyYAML as an optional extra for YAML configs).

# Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author used Anthropic's Claude (via the
Claude Code assistant) to draft and to improve the language and readability of
the manuscript, and to assist with the implementation and validation of the
accompanying software. The research question, the software design, and the
conclusions are the author's own. After using these tools, the author reviewed
and edited the content as needed and takes full responsibility for the content
of the publication.

# References

- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
  *The Journal of Portfolio Management*, 40(5), 94–107.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The
  Probability of Backtest Overfitting. *Journal of Computational Finance*,
  20(4), 39–69.
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *The Journal of Portfolio
  Management*, 42(1), 13–28.
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts
  Journal*, 58(4), 36–52.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
