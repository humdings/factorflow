# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The package version is [semantic versioning](https://semver.org/).
`0.1.0` is the version in the tree. It is not tagged and not on PyPI yet.
The public API may change before 1.0.

## [0.1.0] - not yet tagged

Public release of the engine. Tag `v0.1.0` when publishing. Until 1.0, names and behavior may still change.

### Added

- Batch and live engines that agree on sealed rows.
- Clock-time bars (`BarGrid`, including exchange calendars and a 24/7 clock) and information-driven bars (tick, volume, dollar), with as-of sampling onto a shared clock.
- Term graph: `Factor`, `Filter`, `Classifier`, arithmetic, rolling and cross-sectional operators, `Pipeline`.
- `factorflow.sim`: a seeded synthetic market, replay against the batch engine, and validation helpers.
- README, contributing guide, NumPy-style docstrings, and an API reference (`mkdocs` / `mkdocstrings`).
- `examples/quickstart.py` on synthetic data.
- `pip install .` extras for `dev`, `notebook`, and `docs`, alongside the uv dependency groups.
- GitHub Actions: tests on Python 3.11–3.13 (Ubuntu and macOS), a Nix check, a Pages docs deploy, and a PyPI publish workflow that uses trusted publishing.

## [0.0.1] - 2026-11-02

Internal version of the same engine, before this public tree. Superseded by 0.1.0.
