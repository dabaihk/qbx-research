# Copyright 2026 The qbx-research Authors.
# SPDX-License-Identifier: Apache-2.0
"""A small, safe expression language for filtering parameter candidates.

Sweeps routinely need to express relationships *between* parameters — a fast
moving-average period must stay below a slow one, a stop must sit below a
target. Rather than ``eval`` arbitrary Python (an injection footgun), we walk a
whitelisted subset of the ``ast`` grammar: names, numeric/boolean literals,
comparisons, boolean connectives, and the four arithmetic operators.

    >>> evaluate("fast < slow and risk <= 0.5", {"fast": 5, "slow": 20, "risk": 0.3})
    True
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

__all__ = ["evaluate", "ConstraintError"]


class ConstraintError(ValueError):
    """Raised when an expression is malformed or references unknown names."""


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_COMPARES = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def evaluate(expression: str, variables: Mapping[str, Any]) -> Any:
    """Safely evaluate ``expression`` against a mapping of variable values.

    Dotted names (``fast.window``) are supported by rewriting them to
    underscores, so a parameter named ``fast.window`` is referenced as
    ``fast_window`` inside the expression — or by its original dotted form,
    whichever the author finds clearer.
    """
    env = {str(key).replace(".", "_"): value for key, value in variables.items()}
    rewritten = str(expression)
    # Longest keys first so that "a.b.c" is rewritten before "a.b".
    for key in sorted((str(k) for k in variables), key=len, reverse=True):
        rewritten = rewritten.replace(key, key.replace(".", "_"))
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ConstraintError(f"could not parse constraint {expression!r}: {exc}") from exc
    return _eval(tree.body, env)


def _eval(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ConstraintError(f"unknown constraint variable: {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not bool(_eval(node.operand, env))
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            value = _eval(node.operand, env)
            return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BoolOp):
        values = [_eval(value, env) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(value) for value in values)
        return any(bool(value) for value in values)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval(comparator, env)
            try:
                ok = _COMPARES[type(op)](left, right)
            except KeyError as exc:  # pragma: no cover - defensive
                raise ConstraintError("unsupported comparison operator") from exc
            if not ok:
                return False
            left = right
        return True
    raise ConstraintError(f"unsupported expression node: {type(node).__name__}")
