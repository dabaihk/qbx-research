# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""A small, clean-room vectorised backtester.

Turns a series of *target positions* over a price series into a *net return*
series, under an explicit, configurable :class:`BacktestPolicy` — fees,
slippage, fill timing, shorting and leverage. The accounting is deliberately
transparent and standard; it copies no proprietary engine.

The library's selection statistics never require this module — you may bring
your own returns. It exists so strategies that think in positions don't have to
re-derive cost accounting by hand, and so that accounting is *policy-driven* and
reproducible from a YAML file.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import _yaml

__all__ = ["BacktestPolicy", "BacktestResult", "simulate"]

_FILL_MODES = ("next_bar", "same_bar")


@dataclass(frozen=True)
class BacktestPolicy:
    """Execution assumptions for a backtest — the configurable "BT policy".

    Every field is overridable from a dict or a YAML file, so the same strategy
    can be re-scored under different cost and execution regimes without touching
    code.

    Attributes
    ----------
    fee_bps_per_side:
        Commission per side, in basis points of traded notional.
    slippage_bps_per_side:
        Modelled slippage per side, in basis points.
    fill:
        ``"next_bar"`` (act on the prior bar's signal — no look-ahead) or
        ``"same_bar"`` (fill on the signal bar; optimistic).
    allow_short:
        If ``False``, negative target positions are clipped to zero.
    max_leverage:
        Positions are clipped to ``[-max_leverage, +max_leverage]``.
    initial_equity:
        Starting capital for the equity curve.
    periods_per_year:
        Annualisation factor for Sharpe and volatility (e.g. 252 daily).
    annualize:
        If ``False``, Sharpe/volatility are reported per period.
    """

    fee_bps_per_side: float = 1.0
    slippage_bps_per_side: float = 0.5
    fill: str = "next_bar"
    allow_short: bool = True
    max_leverage: float = 1.0
    initial_equity: float = 1.0
    periods_per_year: float = 252.0
    annualize: bool = True

    def __post_init__(self) -> None:
        if self.fill not in _FILL_MODES:
            raise ValueError(f"fill must be one of {_FILL_MODES}, got {self.fill!r}")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        for bps in (self.fee_bps_per_side, self.slippage_bps_per_side):
            if bps < 0:
                raise ValueError("cost basis points must be >= 0")

    @property
    def cost_rate_per_turnover(self) -> float:
        """Fraction of notional charged per unit of position turnover."""
        return (float(self.fee_bps_per_side) + float(self.slippage_bps_per_side)) / 1e4

    # -- (de)serialisation ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BacktestPolicy:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown BacktestPolicy keys: {sorted(unknown)}")
        return cls(**{key: data[key] for key in data})

    @classmethod
    def from_yaml(cls, path: str | Path) -> BacktestPolicy:
        """Load a policy from a YAML file (requires the ``yaml`` extra)."""
        return cls.from_dict(_yaml.load_file(path))

    @classmethod
    def from_yaml_string(cls, text: str) -> BacktestPolicy:
        return cls.from_dict(_yaml.load_string(text))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """The outcome of :func:`simulate`.

    ``returns`` and ``equity`` are aligned to the input price index. The scalar
    fields are convenience summaries; :meth:`to_metrics` exposes them as a flat
    mapping ready for :func:`qbx_research.objective.rank`.
    """

    returns: pd.Series
    equity: pd.Series
    sharpe: float
    total_return: float
    max_drawdown_pct: float
    volatility: float
    turnover: float
    n_trades: int

    def to_metrics(self) -> dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown_pct": self.max_drawdown_pct,
            "volatility": self.volatility,
            "turnover": self.turnover,
            "n_trades": float(self.n_trades),
        }


def simulate(
    prices: Sequence[float] | pd.Series,
    position: Sequence[float] | pd.Series,
    *,
    policy: BacktestPolicy | None = None,
) -> BacktestResult:
    """Backtest ``position`` over ``prices`` under ``policy``.

    Parameters
    ----------
    prices:
        Price level per period (the time axis).
    position:
        Target position per period, signed; ``+1`` = fully long, ``-1`` = fully
        short. Clipped per the policy's ``allow_short`` and ``max_leverage``.
    policy:
        Execution assumptions; defaults to :class:`BacktestPolicy()`.

    Returns
    -------
    BacktestResult
    """
    policy = policy or BacktestPolicy()
    price = pd.to_numeric(pd.Series(prices), errors="coerce").astype(float)
    target = pd.to_numeric(pd.Series(position), errors="coerce").reindex(price.index)
    target = target.fillna(0.0).astype(float)

    if not policy.allow_short:
        target = target.clip(lower=0.0)
    target = target.clip(-policy.max_leverage, policy.max_leverage)

    # Fill timing: next_bar acts on the prior bar's signal (no look-ahead).
    held = target.shift(1).fillna(0.0) if policy.fill == "next_bar" else target

    market_return = price.pct_change().fillna(0.0)
    gross = held * market_return
    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * policy.cost_rate_per_turnover
    net = (gross - costs).rename("return")

    equity = (policy.initial_equity * (1.0 + net).cumprod()).rename("equity")

    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    mean = float(net.mean())
    scale = float(np.sqrt(policy.periods_per_year)) if policy.annualize else 1.0
    sharpe = round(mean / std * scale, 10) if std > 0 else 0.0
    drawdown = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    total_return = float(equity.iloc[-1] / policy.initial_equity - 1.0) if len(equity) else 0.0

    return BacktestResult(
        returns=net,
        equity=equity,
        sharpe=sharpe,
        total_return=round(total_return, 12),
        max_drawdown_pct=round(abs(drawdown), 12),
        volatility=round(std * scale, 12),
        turnover=round(float(turnover.sum()), 12),
        n_trades=int((turnover > 1e-12).sum()),
    )
