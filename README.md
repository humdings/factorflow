# factorflow

A Pipeline-style factor, filter and classifier engine that runs on a **bar grid** (daily, minute or any fixed frequency). It has two executors that are guaranteed to agree:

- **Batch** for research and backtests over a whole history.
- **Incremental** for live use. Ticks update the **open bar**, the last and still provisional row. `revise()` recomputes only that row, and `seal()` closes the bar. Sealed rows are bit-for-bit identical to the batch result. A revision is identical to a batch run that treats the open values as final.

Every live value carries an **open/sealed** flag.

```python
import factorflow as ff

grid = ff.BarGrid("XNYS", "1min")
close, volume = ff.BarData.close, ff.BarData.volume
sector = ff.Column("sector", ff.Classifier)

ret = close.pct_change(1)
universe = (close * volume).rolling_mean(30).top(100)
score = close.pct_change(30).zscore(groupby=sector, mask=universe) - ret.rolling_std(30).rank(mask=universe)
pipe = ff.Pipeline({"score": score}, screen=universe)

panel = ff.Panel.from_frame(bars_df, grid)            # history, sealed rows
batch = ff.BatchEngine().run(pipe, panel)             # every row
live = ff.LiveEngine(pipe, panel, grid)               # warmed from the same history
live.tick("AAPL", ts, price=231.4, size=100)          # updates the open bar
snap = live.revise()                                  # snap.is_open is True
live.advance(now)                                     # seals once now >= bar end + grace
```

Status: **M0** (engine core, numpy only). It has no data loaders or broker integration. Those live in the consuming application (humdesk).

- Design: [`docs/DESIGN.md`](docs/DESIGN.md), the time model is in §11.
- Development: `uv sync && uv run pytest && uv run ruff check`. With Nix: `nix develop`, `nix build`.

License: Apache-2.0.
