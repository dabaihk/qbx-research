# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""Declarative description of a parameter search space.

A :class:`Param` is one tunable axis. It is specified in exactly one of two
ways:

* an explicit list of ``values`` (categorical / enumerated), or
* a numeric ``low``/``high`` range with an optional ``step``, ``dtype`` and
  ``scale`` (``"linear"`` or ``"log"``).

A :class:`SearchSpace` is an ordered collection of params. Both are frozen,
hashable value objects — two spaces built the same way compare equal, which
makes sweeps reproducible and cacheable.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Param", "SearchSpace"]

_MAX_RANGE_POINTS = 100_000


@dataclass(frozen=True)
class Param:
    """A single tunable parameter.

    Prefer the :meth:`values` and :meth:`range` constructors over the raw
    initialiser — they document intent and validate eagerly.
    """

    name: str
    values: tuple[Any, ...] | None = None
    low: float | None = None
    high: float | None = None
    step: float | None = None
    dtype: str = "float"  # "int" | "float"
    scale: str = "linear"  # "linear" | "log"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Param.name must be a non-empty string")
        has_values = self.values is not None
        has_range = self.low is not None and self.high is not None
        if has_values == has_range:
            raise ValueError(
                f"Param {self.name!r}: provide exactly one of `values` or `low`/`high`"
            )
        if has_range and self.high < self.low:  # type: ignore[operator]
            raise ValueError(f"Param {self.name!r}: high must be >= low")
        if self.scale not in ("linear", "log"):
            raise ValueError(f"Param {self.name!r}: scale must be 'linear' or 'log'")
        if self.dtype not in ("int", "float"):
            raise ValueError(f"Param {self.name!r}: dtype must be 'int' or 'float'")
        if self.scale == "log" and has_range and (self.low <= 0):  # type: ignore[operator]
            raise ValueError(f"Param {self.name!r}: log scale requires low > 0")

    # -- ergonomic constructors -------------------------------------------------
    @classmethod
    def values_of(cls, name: str, values: Sequence[Any]) -> Param:
        """A categorical axis enumerated by ``values``."""
        if not values:
            raise ValueError(f"Param {name!r}: values cannot be empty")
        return cls(name=name, values=tuple(values))

    @classmethod
    def range(
        cls,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        dtype: str = "float",
        scale: str = "linear",
    ) -> Param:
        """A numeric axis spanning ``[low, high]``."""
        return cls(name=name, low=low, high=high, step=step, dtype=dtype, scale=scale)

    # -- materialisation --------------------------------------------------------
    def grid(self) -> list[Any]:
        """The enumerated grid of values for exhaustive search."""
        if self.values is not None:
            return list(self.values)
        return self._range_values()

    def at(self, unit: float) -> Any:
        """Map a quantile ``unit`` in ``[0, 1)`` onto a concrete value.

        Used by random and Latin-hypercube sampling. For categorical params the
        unit indexes the value list; for numeric params it interpolates across
        the (optionally log-scaled, optionally stepped) range.
        """
        unit = min(max(float(unit), 0.0), 1.0)
        if self.values is not None:
            idx = min(int(unit * len(self.values)), len(self.values) - 1)
            return self.values[idx]
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        if self.scale == "log":
            raw = math.exp(math.log(lo) + unit * (math.log(hi) - math.log(lo)))
        else:
            raw = lo + unit * (hi - lo)
        return self._snap(raw)

    def _range_values(self) -> list[Any]:
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        if self.scale == "log":
            # A modest, well-spaced log grid; callers wanting fine control pass values.
            count = 5
            points = [
                math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / (count - 1))
                for i in range(count)
            ]
            return [self._snap(point) for point in points]
        step = self.step
        if step is None:
            step = 1 if self.dtype == "int" else (hi - lo) / 4 if hi != lo else 1
        step = float(step)
        if step <= 0:
            raise ValueError(f"Param {self.name!r}: step must be > 0")
        values: list[Any] = []
        current, guard = lo, 0
        while current <= hi + step / 1_000_000:
            values.append(self._snap(current))
            current += step
            guard += 1
            if guard > _MAX_RANGE_POINTS:
                raise ValueError(f"Param {self.name!r}: range produced too many values")
        return values

    def _snap(self, raw: float) -> Any:
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        if self.step:
            raw = lo + round((raw - lo) / float(self.step)) * float(self.step)
        raw = min(max(raw, lo), hi)
        return int(round(raw)) if self.dtype == "int" else round(raw, 10)


@dataclass(frozen=True)
class SearchSpace:
    """An ordered set of :class:`Param` axes with unique names."""

    params: tuple[Param, ...]

    def __init__(self, params: Iterable[Param]) -> None:
        ordered = tuple(params)
        names = [param.name for param in ordered]
        if len(names) != len(set(names)):
            raise ValueError("SearchSpace param names must be unique")
        # Sort by name for a canonical, order-independent identity.
        object.__setattr__(self, "params", tuple(sorted(ordered, key=lambda p: p.name)))

    def __iter__(self):
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    @property
    def names(self) -> list[str]:
        return [param.name for param in self.params]

    def grid_size(self) -> int:
        """Number of points an exhaustive grid sweep would visit."""
        size = 1
        for param in self.params:
            size *= len(param.grid())
        return size
