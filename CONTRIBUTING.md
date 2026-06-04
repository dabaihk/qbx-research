# Contributing to qbx-research

Thanks for your interest in improving `qbx-research`. This document explains how
to contribute code, report issues, and get support.

## Reporting issues

Please open an issue at
<https://github.com/dabaihk/qbx-research/issues> and include:

- what you expected to happen and what actually happened,
- a minimal, reproducible example (input data shape, the call you made, the
  output or traceback),
- your Python, NumPy, and pandas versions, and the `qbx-research` version.

For suspected statistical/numerical errors, please cite the formula or reference
you are comparing against — the selection metrics follow published methods
(Bailey & López de Prado, Lo) and we aim to match them.

## Asking for help

For usage questions, open a
[GitHub Discussion](https://github.com/dabaihk/qbx-research/discussions) or an
issue labelled `question`.

## Development setup

```bash
git clone https://github.com/dabaihk/qbx-research.git
cd qbx-research
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```bash
pytest          # the full test suite must pass
ruff check .    # lint must be clean
```

## Pull requests

1. Fork the repo and create a topic branch.
2. Keep changes focused; one logical change per PR.
3. Add or update tests for any behaviour change. Numerical functions should have
   tests that pin expected values or invariants.
4. Keep the public API documented (docstrings) and the README accurate.
5. Ensure `pytest` and `ruff check .` pass, then open the PR with a clear
   description of the motivation and approach.

## Design principles

- **Dependency-light:** runtime dependencies are limited to NumPy and pandas.
  Please do not add heavy dependencies (e.g. SciPy) without discussion.
- **Separation of concerns:** the selection statistics are pure functions of
  returns; they must not depend on any particular backtesting engine or data
  source.
- **Correctness first:** implementations follow published formulas and are
  validated numerically. Changes to the statistics must preserve that.

## License

By contributing, you agree that your contributions will be licensed under the
project's [Apache License 2.0](LICENSE).
