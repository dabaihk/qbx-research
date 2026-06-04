# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic JSON canonicalisation for content hashing and de-duplication.

A single source of truth for turning arbitrary (possibly numpy/pandas-flavoured)
values into a stable string, so that two parameter dictionaries that are *equal*
always hash to the same key — regardless of insertion order or numeric dtype.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

__all__ = ["canonical_json"]


def _json_ready(value: Any) -> Any:
    """Coerce a value into JSON-serialisable, canonical form.

    Numpy scalars collapse to their Python counterparts, non-finite floats and
    missing values collapse to ``None``, and timestamps render as ISO-8601. The
    function is import-light: numpy/pandas are only consulted if present.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value

    # Optional numeric stacks — degrade gracefully when unavailable.
    try:  # pragma: no cover - exercised only when numpy is installed
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            fvalue = float(value)
            return None if (math.isnan(fvalue) or math.isinf(fvalue)) else fvalue
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:  # pragma: no cover
        pass

    try:  # pragma: no cover - exercised only when pandas is installed
        import pandas as pd

        if value is pd.NA or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except (ImportError, TypeError, ValueError):  # pragma: no cover
        pass

    return value


def canonical_json(value: Any) -> str:
    """Return a stable, sorted, whitespace-free JSON encoding of ``value``.

    >>> canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    True
    """
    return json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
