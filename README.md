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

Status: **M0** (engine core, numpy only). It has no data loaders or broker integration. Those
live in the consuming application (humdesk).

- Design: [`docs/DESIGN.md`](docs/DESIGN.md) (the time model is in §11, M0 as built in §12);
  bars and time paradigms: [`docs/BARS.md`](docs/BARS.md).
- Development: `uv sync && uv run pytest && uv run ruff check`. Benchmark:
  `uv run python benchmarks/bench_minute.py`. With Nix: `nix develop`, `nix build`, `nix flake check`.

License: Apache-2.0.
