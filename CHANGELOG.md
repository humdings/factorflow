# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The package version is [semantic versioning](https://semver.org/).
`0.1.0` is the version in the tree. It is not tagged and not on PyPI yet.
The public API may change before 1.0.

## [0.1.0] - not yet tagged

Public release of the engine. Tag `v0.1.0` when publishing. Until 1.0, names and behavior may still change.

### Added

- `Factor.exp`, the natural exponential, on the same full-row path as `log`.
- Rolling `median`, `quantile`, `skew`, `kurtosis`, `mad`, `count`, and `product`. A window that contains a NaN is NaN, same as the other rolling reductions. `quantile` takes a fraction in `[0, 1]`.
- Cross-sectional `mean`, `sum`, `median`, `std`, `var`, `min`, `max`, `quantile`, `skew`, `kurtosis`, `mad`, `count`, and `product`. They take `mask` and `groupby`, leave missing values out of the group, and broadcast the statistic onto the assets that entered it. `std` and `var` are the sample versions (divisor `count - 1`), matching `rolling_std` and `zscore`.
- `ewm_std`, with the same half-life and missing-value rule as `ewm_mean`. The deviation is NaN until two finite observations have been seen. After that it is the square root of the bias-included exponentially weighted variance.
- Cumulative `cumsum`, `cumprod`, `cummin`, and `cummax`. A missing observation keeps the running value, and the run starts at the first finite observation.
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
