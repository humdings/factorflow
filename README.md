# factorflow

A Pipeline-style factor, filter and classifier engine that runs on **bars**: clock-time bars
(daily, minute or any fixed frequency on an exchange calendar or a 24/7 clock) or
information-driven bars (tick, volume and dollar bars). It has two executors that are
guaranteed to agree:

- **Batch** for research and backtests over a whole history.
- **Incremental** for live use. Ticks update the **open bar**, the last and still provisional row.
  `revise()` recomputes only that row, and `seal()` closes the bar. Sealed rows are bit-for-bit
  identical to the batch result. A revision is identical to a batch run that treats the open
  values as final.

Every live value carries an **open/sealed** flag (`Snapshot.is_open`, and an `is_open` column in frames).

```python
import factorflow as ff

grid = ff.BarGrid("XNYS", "1min")  # 2 s grace before a minute bar seals
close, volume = ff.BarData.close, ff.BarData.volume
sector = ff.Column("sector", ff.Classifier)

ret = close.pct_change(1)
universe = (close * volume).rolling_mean(30).top(100)
score = close.pct_change(30).zscore(groupby=sector, mask=universe) - ret.rolling_std(30).rank(mask=universe)
pipe = ff.Pipeline({"score": score}, screen=universe)

starts, ends = grid.bars("2026-10-01", "2026-10-30")
panel = ff.build_time_bars(trades, assets, starts, ends)  # or ff.Panel.from_frame(bars_df)
panel.columns["sector"] = sector_codes  # (T, N) int codes
batch = ff.BatchEngine().run(pipe, panel)  # every row; batch.frame() -> polars
live = ff.LiveEngine(pipe, panel, grid, observers=[print])  # warmed from the same history
live.tick("AAPL", ts, price=231.4, size=100)  # updates the open bar
snap = live.revise()  # snap.is_open is True
live.advance(now)  # seals once now >= bar end + grace
```

Information-driven bars ([`docs/BARS.md`](docs/BARS.md)):

```python
builder = ff.EventBarBuilder("dollar", ff.AdaptiveDaily(bars_per_day=50, lookback_days=20, initial=5e6),
                             overshoot="close")
bars, open_bar = ff.build_event_bars(ts, price, size, "dollar", builder.rule)  # batch, one asset
ep = ff.EventPipeline(pipe_ts, builder, history=bars, asset="BTC")             # per-asset, live
snaps = ep.update(ts, price, size)            # sealed snapshot per closed bar + open revision
clock = ff.sample_panel({"BTC": (bars["end"], {"vol": vol_btc}), ...}, grid_ends)  # as-of sampling
ff.BatchEngine().run(ff.Pipeline({"rk": ff.Column("vol").rank()}), clock)          # cross-section
```

Status: **M0** (engine core, numpy only) plus a simulation and replay harness (`factorflow.sim`).
It has no data loaders or broker integration. Those live in the consuming application (humdesk).

- Design: [`docs/DESIGN.md`](docs/DESIGN.md) (the time model is in §11, M0 as built in §12, the simulator in §13);
  bars and time paradigms: [`docs/BARS.md`](docs/BARS.md).

## Quickstart: the simulated market

`factorflow.sim` generates a seeded synthetic market with **planted ground truth**, so the
engine can be exercised and checked end to end without any data feed:

- equities on XNYS (including early closes such as 2026-11-27, 13:00 ET), with market beta,
  size/value/quality styles, industries (a `Classifier`), GARCH vol and Student-t shocks;
- BTC/ETH/SOL trading 24/7;
- Poisson trades with a U-shaped intraday profile, bid/ask bounce, illiquid names with missing
  bars, halts and overnight gaps;
- a planted alpha signal with a known IC (0.03).

```bash
uv sync
uv run python -m factorflow.sim demo        # simulate, verify live == batch, validate, then a live table at 60x
uv run python -m factorflow.sim validate    # factor / IC / backtest / bar-normality report (~5 s)
uv run python -m factorflow.sim bench --names 100 500 --minutes 30
uv run --group notebook jupyter lab examples/demo.ipynb   # executed notebook with plots
```

`demo` options: `--names N`, `--seed S`, `--speed X` (market seconds per wall second), `--seconds`
(how long the live table runs), `--top`, `--skip-validation`. The CLI exits non-zero if any live row
differs from batch.

```python
import factorflow as ff
from factorflow.sim import SimConfig, simulate
from factorflow.sim.pipelines import minute_pipeline
from factorflow.sim.replay import replay, verify_replay
from factorflow.sim.validate import validate

mkt = simulate(SimConfig(seed=7).scaled(n=100))  # mkt.equities / mkt.crypto: AssetSim
eq = mkt.equities  # .daily Panel, .trades, .truth, .halts
panel, _ = eq.time_bars("1min")  # clock bars built from the simulated ticks
h = 390  # warm on the first session (2026-11-25), replay the rest
v = verify_replay(
    minute_pipeline(),
    panel,
    eq.trades,
    ff.BarGrid("XNYS", "1min"),
    h,
    extra_columns=("signal", "industry"),
    check_open_every=50,
)
assert v.ok  # every sealed row == BatchEngine, bit for bit
print(validate(mkt).report())  # recovered factors, IC, decile long-short, kurtosis
```

`replay(engine, trades, speed=60.0, on_revise=..., on_seal=...)` streams ticks through a `LiveEngine`
either flat out or in accelerated real time. In accelerated mode, bars with no trades still seal on the clock.

Results with seed 7, 100 names and 252 days:

- **Factor recovery:** market corr 0.91 with slope 1.01; styles corr 0.81 to 0.89.
- **Rolling beta vs true beta:** corr 0.91.
- **Residual IC:** 0.033 ± 0.007 vs 0.030 planted.
- **Decile long-short:** 20 bp/day, Sharpe 3.7.
- **Excess kurtosis of intraday returns:** 1-minute bars 3.7, 5-minute 1.7, volume bars 0.27, dollar bars 0.36.

Replay performance at 500 names on the demo minute pipeline (36 nodes):

- **Ingest:** about 520k ticks/s.
- **Revise latency:** about 0.7 ms.
- **Seal latency:** about 0.35 ms.

See [`docs/DESIGN.md`](docs/DESIGN.md) §13.

## Development

`uv sync && uv run pytest && uv run ruff check && uv run pyright`. Older micro-benchmark:
`uv run python benchmarks/bench_minute.py`. With Nix: `nix develop`, `nix build`, `nix flake check`.

CI is not enabled yet. The GitHub Actions workflow is in [`docs/ci-workflow.yml`](docs/ci-workflow.yml).
Copy it to `.github/workflows/ci.yml` with a token that has the `workflow` scope.

License: Apache-2.0.
