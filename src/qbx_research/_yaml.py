# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""Optional YAML loading.

PyYAML is an optional dependency: the config dataclasses are fully usable from
plain Python dicts, and YAML is only needed if you choose to load from files.
These helpers raise a clear, actionable error when it is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load_file", "load_string"]

_INSTALL_HINT = (
    "PyYAML is required to load YAML configs. Install it with "
    "`pip install 'qbx-research[yaml]'` (or `pip install pyyaml`)."
)


def _yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without pyyaml
        raise ImportError(_INSTALL_HINT) from exc
    return yaml


def load_file(path: str | Path) -> dict[str, Any]:
    """Parse a YAML file into a dict (empty file -> empty dict)."""
    text = Path(path).read_text(encoding="utf-8")
    return load_string(text)


def load_string(text: str) -> dict[str, Any]:
    """Parse a YAML string into a dict."""
    data = _yaml().safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must define a mapping at the top level")
    return data
