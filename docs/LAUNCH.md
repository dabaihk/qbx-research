# qbx-research — Launch & Amplification Checklist

A living to-do list for getting `qbx-research` in front of the right people.
Check items off as you go. Ordered so each step strengthens the landing for the
next. The core message everywhere: **lead with the problem (your backtest's
Sharpe is inflated because you selected it), not the package.**

---

## Phase 0 — Polish the landing (do before any launch)

- [ ] Add one figure to the README (Sharpe-inflation curve, or in-sample-vs-OOS
      scatter) — a chart 3–5×'s social sharing. Commit a seeded script to `docs/`.
- [ ] Verify `pip install qbx-research` works in a clean venv and quickstart runs
      in < 60s. (Currently: ✅ verified.)
- [ ] Confirm README math renders on GitHub. (Currently: ✅ `math` fenced blocks.)
- [ ] Add GitHub repo **topics**: `deflated-sharpe-ratio`, `backtesting`,
      `overfitting`, `pbo`, `sharpe-ratio`, `quant`, `algorithmic-trading`.
- [ ] Add a short "Cite this" section once the JOSS DOI exists.

## Phase 1 — Durable discovery (low effort, long tail)

- [ ] PR to **awesome-quant** (wilsonfreitas/awesome-quant) under risk/backtesting.
- [ ] PR to **awesome-systematic-trading** and **awesome-python** (finance section).
- [ ] Submit to newsletters: **Python Weekly**, **PyCoder's Weekly**,
      **Awesome Python Newsletter**, **Quant Insider**.
- [ ] Add to **Libraries.io** / ensure good PyPI classifiers & keywords. (PyPI: ✅ live.)

## Phase 2 — Papers (credibility anchors; run in parallel, ~2 wk to land)

- [ ] **JOSS** submission — `paper/paper.md` + `paper/paper.bib` drafted (✅).
      - [ ] Fill author ORCID + confirm affiliation.
      - [ ] Tag a release (e.g. `v0.1.0`) and archive on **Zenodo** → get a DOI.
      - [ ] Submit at joss.theoj.org; respond to reviewers on GitHub.
- [ ] **SSRN** whitepaper — `docs/whitepaper.md` drafted (✅).
      - [ ] Convert to PDF (`pandoc docs/whitepaper.md -o whitepaper.pdf`).
      - [ ] Post to SSRN (q-fin / Computational Finance eJournal).
- [ ] (Optional) **arXiv q-fin** — ONLY if you add a real contribution
      (reproducibility/benchmark study of Sharpe inflation, or an effective-trials
      comparison). A bare re-implementation is not worth an arXiv post. Note:
      q-fin may require endorsement for first-time submitters.

## Phase 3 — High-signal launches (one-shot; need Phase 0 done first)

- [ ] **Show HN** — title e.g. *"Show HN: qbx-research – deflate your backtest's
      Sharpe for the search you did to find it."* Be present to answer scrutiny.
- [ ] **r/algotrading** post (lead with the demo output: DSR 0.88, PBO 0.21).
- [ ] **r/quant**, **r/quantfinance**, **r/Python** ("I made this") — space these out.
- [ ] **X/fintwit** thread (#quant #algotrading) with one chart.
- [ ] **LinkedIn** post under your own name (prop-firm provenance = credibility).

## Phase 4 — Slow-burn authority

- [ ] **Wilmott forums**, **Elite Trader**, **QuantConnect community** intros.
- [ ] A short blog/dev.to post: "Why your backtest Sharpe is a lie (and the math
      to fix it)" — cross-post; link to repo + whitepaper.

---

## Monitoring: Quantitative Finance Stack Exchange (recurring)

Goal: **build durable authority** by answering real questions well and
referencing `qbx-research` only where it genuinely helps. Do **not** drop links
into unrelated threads — Quant SE punishes self-promotion and rewards substance.

- [ ] Create a saved search / RSS for these **tags** on quant.stackexchange.com:
      - [ ] [`sharpe-ratio`](https://quant.stackexchange.com/questions/tagged/sharpe-ratio)
      - [ ] [`backtesting`](https://quant.stackexchange.com/questions/tagged/backtesting)
      - [ ] [`overfitting`](https://quant.stackexchange.com/questions/tagged/overfitting)
      - [ ] [`performance-evaluation`](https://quant.stackexchange.com/questions/tagged/performance-evaluation)
      - [ ] [`statistics`](https://quant.stackexchange.com/questions/tagged/statistics) (filter for selection-bias / multiple-testing)
      - [ ] [`hypothesis-testing`](https://quant.stackexchange.com/questions/tagged/hypothesis-testing)
- [ ] RSS feed per tag: `https://quant.stackexchange.com/feeds/tag/<tag>` —
      subscribe in your reader so new questions surface automatically.
- [ ] Keyword watch (Stack Exchange search): `deflated sharpe`, `probability of
      backtest overfitting`, `PBO`, `CSCV`, `probabilistic sharpe`, `selection
      bias backtest`, `expected maximum sharpe`, `multiple testing trading`.
- [ ] **Cadence:** scan ~2×/week (15 min). Answer 1–2 questions where DSR/PSR/PBO
      or effective-trials is the right tool. Show the math first; mention the lib
      as "a small reference implementation" with a one-line code snippet.
- [ ] Track wins: log Q&A links + upvotes/accepts here:
      - [ ] _(answer link)_ — _(topic)_
      - [ ] _(answer link)_ — _(topic)_
- [ ] Cross-pollinate: when an answer is strong, the same content (lightly
      reframed) often fits an r/quant or blog post.

> Etiquette: a great Quant SE answer that *happens* to use the library converts
> skeptical experts into stars far better than any direct promotion.

---

## Quick metrics to watch (optional)

- [ ] GitHub stars / forks trend
- [ ] PyPI downloads (`pypistats.org/packages/qbx-research`)
- [ ] JOSS DOI citations (Google Scholar alert on the paper title)
- [ ] Inbound issues / discussions (a sign of real adoption)
