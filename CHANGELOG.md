# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The package version is [semantic versioning](https://semver.org/).
Nothing has been published to PyPI yet. `0.0.1` is the version in the tree.

## [Unreleased]

### Changed

- Public documentation: README, CONTRIBUTING, this changelog, `examples/quickstart.py`,
  and an API reference built from NumPy-style docstrings (`mkdocs` / `mkdocstrings`).
- Removed the internal design proposal (`docs/DESIGN.md`). Bar rules stay in `docs/BARS.md`.
- GitHub Actions workflow lives at `.github/workflows/ci.yml`. It runs ruff (including
  pydocstyle), interrogate, pytest, the quickstart, and `mkdocs build --strict`.
- `pip install -e ".[dev]"`, `pip install -e ".[notebook]"`, and `pip install -e ".[docs]"`
  extras, alongside the existing uv dependency groups.

The Python API is unchanged.

## [0.0.1] - 2026-11-02

### Added

- Batch and live engines that agree on sealed rows.
- Clock-time bars (`BarGrid`, including exchange calendars and a 24/7 clock) and information-driven bars (tick, volume, dollar), with as-of sampling onto a shared clock.
- Term graph: `Factor`, `Filter`, `Classifier`, arithmetic, rolling and cross-sectional operators, `Pipeline`.
- `factorflow.sim`: a seeded synthetic market, replay against the batch engine, and validation helpers.
