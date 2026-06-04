# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""Illustrative strategies for documentation, tests and the quickstart.

These exist purely to demonstrate the :class:`~qbx_research.strategy.Strategy`
interface end-to-end. They are intentionally simple, generate their own
synthetic data, and are **not** investment advice or a template for production
research.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestPolicy
from .space import Param, SearchSpace
from .strategy import SignalStrategy

__all__ = ["synthetic_prices", "MovingAverageCrossover"]


def synthetic_prices(
    n: int = 750,
    *,
    seed: int = 0,
    drift: float = 0.0003,
    volatility: float = 0.01,
    start: float = 100.0,
) -> pd.Series:
    """A deterministic geometric-Brownian-motion price path.

    Seeded so demos and tests are reproducible. Returns a price ``pd.Series``
    indexed ``0..n-1``.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.standard_normal(n) * volatility + drift
    return pd.Series(start * np.exp(np.cumsum(log_returns)), name="price")


class MovingAverageCrossover(SignalStrategy):
    """Go long (and optionally short) on a fast/slow simple-moving-average cross.

    A textbook example with exactly the kind of two knobs that invite
    overfitting — perfect for showing what the Deflated Sharpe Ratio and PBO do
    to an in-sample winner. As a :class:`~qbx_research.strategy.SignalStrategy`
    it only defines *positions*; the configurable
    :class:`~qbx_research.backtest.BacktestPolicy` owns costs, fill and shorting.
    """

    name = "moving_average_crossover"

    def __init__(
        self,
        prices: Sequence[float] | pd.Series | None = None,
        *,
        fast_windows: Sequence[int] = (5, 10, 20, 50),
        slow_windows: Sequence[int] = (20, 50, 100, 200),
        policy: BacktestPolicy | None = None,
        seed: int = 0,
    ) -> None:
        self.prices = (
            pd.Series(np.asarray(prices, dtype=float))
            if prices is not None
            else synthetic_prices(seed=seed)
        )
        self.policy = policy or BacktestPolicy(allow_short=False)
        self._fast = tuple(fast_windows)
        self._slow = tuple(slow_windows)

    @property
    def space(self) -> SearchSpace:
        return SearchSpace(
            [
                Param.values_of("fast", self._fast),
                Param.values_of("slow", self._slow),
            ]
        )

    def positions(self, params: Mapping[str, Any]) -> pd.Series:
        fast_ma = self.prices.rolling(int(params["fast"])).mean()
        slow_ma = self.prices.rolling(int(params["slow"])).mean()
        position = pd.Series(0.0, index=self.prices.index)
        position[fast_ma > slow_ma] = 1.0
        position[fast_ma < slow_ma] = -1.0  # policy clips shorts when disallowed
        return position
