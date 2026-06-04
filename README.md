# qbx-research

**Parameter sweeps and selection-aware backtest statistics for quantitative research.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-43_passing-brightgreen)
[![Live Demo](https://img.shields.io/badge/live_demo-qbxwin.com-purple)](https://qbxwin.com)

*An excerpt from our proprietary trading firm's research infrastructure. See part of the wider platform live at **[qbxwin.com](https://qbxwin.com)**.*

> Searching hard for a good strategy and reporting its Sharpe ratio is like
> firing a shotgun at a barn and painting a target around the tightest cluster.
> This library paints the target *first*.

---

## Why this exists

Every backtest you run is a hypothesis test. Run one and a Sharpe of 1.5 means
something. Run **a thousand** — sweeping windows, thresholds, holding periods —
and keep the best, and that same 1.5 means almost nothing: it is the *maximum of
a thousand draws*, and the maximum of pure noise is large. This is **backtest
overfitting**, and it is the dominant reason strategies that look brilliant in
research die in production.

The mechanism is not subtle. If you try $N$ independent strategies that all have
*zero* true skill, the expected best Sharpe you will observe is not zero — it
grows without bound in $N$:

```math
\mathbb{E}\!\left[\max_{i \le N} \widehat{SR}_i\right] \;\approx\;
\sigma_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
+ \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right]
```

where $\sigma_{SR}$ is the spread of Sharpe across trials, $\Phi^{-1}$ is the
inverse standard normal, and $\gamma \approx 0.5772$ is the Euler–Mascheroni
constant. With a hundred trials and a realistic dispersion, the luck-only
benchmark is already a Sharpe near **1.0**. Beat *that*, not zero.

`qbx-research` does two things, and treats them as inseparable:

1. **Search** a parameter space — grid, random, or Latin hypercube — and rank
   the results against an explicit objective.
2. **Deflate** the winner for the very search that produced it — Probabilistic
   and Deflated Sharpe ratios, and the Probability of Backtest Overfitting.

You implement one abstract `Strategy`; the library never sees your data, signals
or execution model — only the returns your backtest produces.

### Why overfit prevention is the whole game

- **A backtest is a maximum, not a sample.** The number you are tempted to
  report is selected *because* it was extreme. Reporting it unadjusted is a
  measurement error, not a judgement call.
- **More effort makes it worse.** The harder you search, the higher the
  luck-only bar climbs — yet a naive Sharpe stays silent about how many trials
  bought it. Diligence, uncorrected, manufactures false confidence.
- **Capital is the loss function.** A strategy accepted on an inflated Sharpe is
  not a missed opportunity — it is a live position drawing down real money. The
  asymmetry is brutal, so the statistics should be conservative by construction.

The right question is never "what was the best Sharpe?" It is **"is the best
Sharpe larger than what this search would have produced from noise alone?"**
That is precisely what the Deflated Sharpe Ratio answers.

---

## Install

```bash
pip install qbx-research
```

Runtime dependencies are just `numpy` and `pandas`. No SciPy — the normal
quantile function ships as a vendored rational approximation. YAML configs are
optional:

```bash
pip install "qbx-research[yaml]"   # adds PyYAML for BacktestPolicy/RunConfig.from_yaml
```

## The 60-second tour

Implement a `Strategy`, then hand it to `evaluate`:

```python
import pandas as pd
from qbx_research import Strategy, Param, SearchSpace, Objective, evaluate

class MovingAverageCrossover(Strategy):
    name = "ma_crossover"

    def __init__(self, prices: pd.Series):
        self.prices = prices

    @property
    def space(self) -> SearchSpace:                 # what to search
        return SearchSpace([
            Param.values_of("fast", [5, 10, 20, 50]),
            Param.values_of("slow", [20, 50, 100, 200]),
        ])

    def backtest(self, params) -> pd.Series:        # how it performs
        fast = self.prices.rolling(params["fast"]).mean()
        slow = self.prices.rolling(params["slow"]).mean()
        position = (fast > slow).astype(float)
        return position.shift(1).fillna(0) * self.prices.pct_change().fillna(0)

report = evaluate(
    MovingAverageCrossover(my_prices),
    objective=Objective.maximize("sharpe"),
    constraints=["fast < slow"],     # relate parameters to one another
)

print(report.best.candidate.params)             # the in-sample winner
print(report.deflated_sharpe.probability)       # ...after deflating for the search
print(report.pbo.pbo)                            # probability the whole search is overfit
```

`evaluate` runs the entire loop — sweep → backtest each candidate → rank →
deflate → estimate overfitting — and only ever sees the returns your `backtest`
returns. A runnable version (with synthetic prices, no data needed) lives in
[`examples/quickstart.py`](examples/quickstart.py); the demo strategy ships in
`qbx_research.demos`.

### Positions, not returns? Use `SignalStrategy` + a policy

Most strategies are easier to express as *positions* than as returns. Subclass
`SignalStrategy`, implement `positions(params)`, and let a configurable
`BacktestPolicy` own the execution assumptions — costs, fill timing, shorting,
leverage — while the included vectorised backtester handles the accounting:

```python
from qbx_research import SignalStrategy, BacktestPolicy, Param, SearchSpace

class MACrossover(SignalStrategy):
    def __init__(self, prices, policy=None):
        self.prices = prices
        self.policy = policy or BacktestPolicy(fee_bps_per_side=1.0, allow_short=False)

    @property
    def space(self):
        return SearchSpace([Param.values_of("fast", [5, 10, 20]),
                            Param.values_of("slow", [50, 100, 200])])

    def positions(self, params):                     # +1 long, -1 short, 0 flat
        fast = self.prices.rolling(params["fast"]).mean()
        slow = self.prices.rolling(params["slow"]).mean()
        pos = (fast > slow).astype(float) - (fast < slow).astype(float)
        return pos                                   # policy clips shorts, applies costs
```

### Configuration from YAML

The backtest policy and the run itself are declarative — keep them in version
control, not in call sites:

```yaml
# policy.yaml — execution assumptions
fee_bps_per_side: 1.0
slippage_bps_per_side: 0.5
fill: next_bar          # next_bar (no look-ahead) | same_bar
allow_short: false
max_leverage: 1.0
periods_per_year: 252
```

```yaml
# run.yaml — everything except the strategy
method: grid            # grid | random | latin_hypercube
constraints: ["fast < slow"]
objective_metric: sharpe
objective_direction: maximize
metric_constraints: [["max_drawdown_pct", "<=", 0.25]]
pbo_folds: 8
```

```python
from qbx_research import BacktestPolicy, RunConfig, evaluate

policy = BacktestPolicy.from_yaml("policy.yaml")
config = RunConfig.from_yaml("run.yaml")
report = evaluate(MACrossover(my_prices, policy=policy), config=config)
```

Explicit keyword arguments to `evaluate` always override the config, so YAML is
a default, never a straitjacket. Example configs live in
[`examples/policy.yaml`](examples/policy.yaml) and
[`examples/run.yaml`](examples/run.yaml).

---

## The mathematics

Notation: a strategy produces $n$ per-period returns with sample mean $\mu$ and
standard deviation $\sigma$. $\Phi$ is the standard normal CDF; $\hat\gamma_3$
and $\hat\gamma_4$ are the sample skewness and kurtosis of the returns.

### 1. The Sharpe ratio is an *estimate*, with error bars

```math
\widehat{SR} = \frac{\mu}{\sigma}
```

A point Sharpe hides its own uncertainty. For non-normal returns its standard
error is (Lo 2002; Mertens 2002):

```math
\sigma_{\widehat{SR}} =
\sqrt{\frac{1 - \hat\gamma_3\,\widehat{SR}
+ \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^{\,2}}{\,n - 1\,}}
```

Negative skew and fat tails *inflate* this error — which is exactly the return
profile of most strategies that sell insurance. A Sharpe with no error bar is a
headline with no story.

### 2. Probabilistic Sharpe Ratio — beating a benchmark, honestly

The **PSR** is the probability that the *true* Sharpe exceeds a benchmark
$SR^\ast$, given the sample's length and shape:

```math
\widehat{PSR}(SR^\ast) =
\Phi\!\left(
\frac{\left(\widehat{SR} - SR^\ast\right)\sqrt{\,n - 1\,}}
{\sqrt{\,1 - \hat\gamma_3\,\widehat{SR}
+ \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^{\,2}\,}}
\right)
```

More data, higher excess Sharpe, and better-behaved tails all push PSR toward 1.
It converts "my Sharpe is 1.5" into "I am 88% confident my Sharpe beats 0.5."

### 3. The selection benchmark — expected maximum Sharpe under the null

Here is the crux. When you keep the best of $N$ trials, the honest benchmark is
**not** zero — it is the largest Sharpe you would expect from $N$ skill-less
trials. Using the order statistics of the normal distribution:

```math
SR_0 = \overline{SR} + \sigma_{SR}
\left[(1 - \gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
+ \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right],
\qquad \gamma \approx 0.5772
```

where $\sigma_{SR}$ is the dispersion of Sharpe *across your candidates* and
$\overline{SR}$ is their mean (taken as $0$ under the strict null). $SR_0$ rises
with both the **number** and the **variance** of trials — search harder, raise
the bar.

### 4. Deflated Sharpe Ratio — the headline number

The **DSR** is simply the PSR measured against that selection-aware benchmark:

```math
\boxed{\;\widehat{DSR} = \widehat{PSR}(SR_0)
= \Phi\!\left(
\frac{\left(\widehat{SR} - SR_0\right)\sqrt{\,n - 1\,}}
{\sqrt{\,1 - \hat\gamma_3\,\widehat{SR}
+ \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^{\,2}\,}}
\right)\;}
```

Read it as: **the probability the strategy has genuine skill, after accounting
for the fact that you went looking for it.** A DSR of 0.95 is the usual
acceptance bar. A glamorous in-sample Sharpe with $\widehat{SR} < SR_0$ deflates
to a DSR below 0.5 — a coin flip dressed as an edge.

### 5. Probability of Backtest Overfitting (PBO) via CSCV

DSR corrects a *number*; PBO interrogates the *process*. Combinatorial Symmetric
Cross-Validation splits the timeline into $S$ slices and, for every way of
choosing $S/2$ of them as in-sample, asks: does the in-sample champion still
look good out-of-sample?

For each split $c$, take the IS winner, find its relative rank $\omega_c \in
(0,1)$ among all candidates **out-of-sample**, and form its logit:

```math
\lambda_c = \ln\!\frac{\omega_c}{1 - \omega_c}
```

A winner that overfits lands below the OOS median ($\omega_c < 0.5$, so
$\lambda_c < 0$). PBO is the share of splits where that happens:

```math
\mathrm{PBO} = \frac{1}{|C|}\sum_{c \in C} \mathbf{1}\!\left[\lambda_c < 0\right]
```

$\mathrm{PBO} \approx 0.5$ means your selection has **no** out-of-sample value —
the in-sample best is a coin flip out of sample. Low PBO means the winner tends
to generalize. Where DSR can be fooled by a single lucky path, PBO resamples the
whole decision.

### 6. Deflating the trial count itself

Two corrections keep the inputs honest:

**Effective number of trials.** Correlated candidates are not independent bets.
From the eigenvalues $\lambda_i$ of the candidate correlation matrix:

```math
N_\text{eff} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}
```

A participation ratio: 100 near-identical sweeps count as a handful of real
trials, not 100.

**Effective sample size.** Autocorrelated returns carry less information than
their length suggests. With autocorrelations $\rho_k$:

```math
n_\text{eff} = \frac{n}{1 + 2\sum_{k=1}^{K} \rho_k^{+}},
\qquad \rho_k^{+} = \max(0, \rho_k)
```

Both feed the formulas above, so neither a redundant grid nor sticky returns can
smuggle in false confidence.

---

## What's inside

### Strategy & harness — `qbx_research.strategy`

`Strategy` is the single abstract base you implement: a `space` property and a
`backtest(params) -> pd.Series` method. `SignalStrategy` is the position-based
variant (implement `positions`; the backtester does the rest).
`evaluate(strategy, ...)` runs the whole research loop and returns a
`StrategyReport` (`ranking`, `returns` matrix, `deflated_sharpe`, `pbo`).
`qbx_research.demos` carries an illustrative `MovingAverageCrossover` on
synthetic prices for docs and tests.

### Backtest & config — `qbx_research.backtest`, `qbx_research.config`

`simulate(prices, positions, *, policy)` is a clean-room vectorised engine —
next-bar/same-bar fill, per-side fees and slippage on turnover, equity curve,
annualised Sharpe, max drawdown, turnover and trade count — returning a
`BacktestResult`. `BacktestPolicy` (execution assumptions) and `RunConfig`
(search method, objective, constraints, folds) are frozen dataclasses that load
from a dict or a YAML file. Nothing here is copied from a proprietary engine.

### Search — `qbx_research.search`

| Method | Behaviour |
| --- | --- |
| `grid` | Full Cartesian product; deterministic, exhaustive. |
| `random` | Uniform draws, de-duplicated by canonical value. |
| `latin_hypercube` | Stratified space-filling; better coverage per sample. |

A `SearchSpace` is an ordered set of `Param` axes — enumerated
(`Param.values_of`) or numeric ranges (`Param.range`, with `step`, `dtype`,
`scale="log"`). `sweep()` returns `Candidate`s with a stable `param_hash`, an
ideal backtest-cache key.

### Objectives — `qbx_research.objective`

`Objective.maximize("sharpe")` with optional hard `Constraint`s and
`with_tie_breaker(...)`. `rank()` orders evaluated candidates, pushing
constraint-violators to the back.

### Selection metrics — `qbx_research.selection`

| Function | Returns | Formula |
| --- | --- | --- |
| `probabilistic_sharpe_ratio` | `float` | §2 — $\widehat{PSR}(SR^\ast)$ |
| `expected_max_sharpe` | `float` | §3 — $SR_0$ |
| `deflated_sharpe_ratio` | `DeflatedSharpe` | §4 — $\widehat{PSR}(SR_0)$ |
| `deflated_sharpe_from_returns` | `DeflatedSharpe` | §4, straight from a return matrix |
| `compute_pbo` | `PBOResult` | §5 — CSCV PBO |
| `effective_num_trials` | `float` | §6 — $N_\text{eff}$ |
| `effective_sample_size` | `float` | §6 — $n_\text{eff}$ |
| `sharpe_standard_error` | `float` | §1 — $\sigma_{\widehat{SR}}$ |

Functions raise `NotComputable(reason)` — a `ValueError` subclass with a
machine-readable `.reason` — when inputs are too thin to support a result.

### Prefer the primitives?

Every step is public, so you can skip the harness and wire it yourself:

```python
from qbx_research import sweep, rank, deflated_sharpe_from_returns, compute_pbo
candidates = sweep(space, method="latin_hypercube", budget=200, seed=1)
# ...evaluate however you like, assemble a return matrix, then:
dsr = deflated_sharpe_from_returns(best_returns, candidate_matrix)
pbo = compute_pbo(candidate_matrix, metric="sharpe", folds=8)
```

---

## Reading the numbers

| Statistic | Healthy | Warning | Reject |
| --- | --- | --- | --- |
| **DSR** $\widehat{PSR}(SR_0)$ | $\ge 0.95$ | $0.90$–$0.95$ | $< 0.90$ |
| **PBO** | $\le 0.10$ | $0.10$–$0.20$ | $> 0.50$ (no OOS value) |
| **PSR** vs $SR^\ast=0$ | $\ge 0.95$ | — | $< 0.90$ |

These are conventions, not laws — calibrate to your own acceptance budget. The
discipline that matters is reporting them *at all*, every time, before capital
moves.

---

## Development

```bash
pip install -e ".[dev]"
pytest          # 34 tests
ruff check .    # lint
python examples/quickstart.py
```

## References

- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.*
  Journal of Portfolio Management, 40(5), 94–107.
- Bailey, Borwein, López de Prado & Zhu (2017). *The Probability of Backtest
  Overfitting.* Journal of Computational Finance, 20(4), 39–69.
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts
  Journal, 58(4), 36–52.

## Provenance & license

This package is an **excerpt from our proprietary trading firm's research
infrastructure** — the parameter-sweep and selection-metrics modules, extracted
and generalised for open release. The statistical implementations match their
in-house source to within $10^{-9}$. The surrounding platform (strategy
modelling, execution, data) remains proprietary and is intentionally kept out of
this repository.

A live demo showcasing part of the project is at **[qbxwin.com](https://qbxwin.com)**.

Released under the [Apache License 2.0](LICENSE); see [`NOTICE`](NOTICE) for
attributions.

## Authorship

Authored by **Yichong Bai** (<yichong.bai@gmail.com>). The open-source extraction,
generalisation and packaging were carried out with [Claude Code](https://claude.com/claude-code)
(Anthropic) under the author's direction.
