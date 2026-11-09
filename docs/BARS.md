# Bars and time paradigms

factorflow computes on **rows**. A row is a bar; the newest row may be *open* (provisional,
revised by ticks) and every other row is *sealed* (immutable). What decides when a bar closes
(the **close trigger**) is separate from the engine. The engine only sees `revise()` and
`seal()`. Two families of triggers are included:

| trigger | bars are | who drives `seal()` | module |
|---|---|---|---|
| clock time | the same for every asset (a grid) | `LiveEngine.tick/advance` from timestamps | `grid.py`, `bars.py` |
| tick count, volume, dollar value | per asset | the `EventBarBuilder` when the threshold is reached | `events.py` |

Imbalance and run bars (AFML §2.3.2) are deferred. They fit the same builder interface: a
different measure plus an expected-value threshold.

## Clock-time bars

* A grid is a calendar × a frequency: `BarGrid("XNYS", "1min")`, `BarGrid("XNYS", "1d")`,
  `BarGrid("24/7", "1min")`. Sessions, holidays and early closes come from `exchange_calendars`.
  The last bar of a session is clipped to the close.
* Bars are half-open `[start, end)` and **labeled by end time**. A trade at `ts` belongs to the
  first bar with `end > ts`.
* **Grace.** A bar seals once a tick or `advance(now)` sees `now >= end + grace`. The default is
  2 s for intraday bars and 0 for daily bars. During the grace period, late prints for the
  bar are still accepted. Trades stamped `>= end` are held and applied to the next bar after
  the seal. Prints for an already-sealed bar are dropped and counted in `late_ticks`.
  There is no corrections path for those late prints yet.
* **Missing bars** carry the close forward: `open = high = low = close = previous close`,
  `volume = dollar = n_ticks = 0`, `has_trade = False` (NaN before an asset's first trade).
  `advance()` seals any number of empty bars.
* Batch (`build_time_bars`) and live (`LiveEngine.tick`) use the same accumulator
  (`BarState`), so live bars equal batch bars when trades arrive in time order within grace.
  `LiveEngine.set_bar()` overwrites an asset's open bar from a broker's in-progress bar.

## Information-driven bars (López de Prado, *Advances in Financial Machine Learning*, ch. 2)

An event bar closes when a running **measure** reaches a **threshold**:

| kind | measure per trade |
|---|---|
| `tick` | 1 |
| `volume` | size |
| `dollar` | price × size |

### Overshoot: the trade that crosses the threshold

`overshoot=` is configurable:

* **`"close"` (default).** The whole crossing trade goes into the bar it completes; the bar
  closes at that trade. Bars contain whole trades (so OHLC, VWAP and trade counts are real
  prints) and `measure >= threshold`. Dropping the crossing trade would leave the bar under
  the threshold. This is the common practical choice. It keeps trade timestamps and counts
  honest, and the overshoot is bounded by one trade.
* **`"split"`.** The trade is cut so that every bar holds exactly `threshold`. The remainder
  starts the next bar, and a very large trade can fill several bars that share its timestamp
  and price. Use this when equal information content per bar matters more than whole trades.
  Tick bars ignore it, since a tick is indivisible.

### Thresholds

* `Fixed(value)`.
* `AdaptiveDaily(bars_per_day, lookback_days=20, initial=..., min_days=1, floor=0)`. The
  threshold for a day is `mean(measure per day over the previous lookback days) / bars_per_day`,
  using **only days completed before that day's first trade** (point-in-time). Days with no
  trades do not count, so weekends are not zero-volume days. Until `min_days` complete days
  exist, `initial` is used. The day is `day_key(ts)`, UTC by default; pass an exchange-session
  key for equities.
* A new threshold takes effect at the first trade of a new day. The open bar is **not** reset
  at day boundaries, and a bar records the threshold that closed it (`threshold` field).

### Guarantees (tested with hypothesis on random trade streams)

* Batch `build_event_bars` equals incremental `EventBarBuilder.update` for any chunking,
  including the open bar. Both run one state machine.
* Thresholds are respected: `close` gives `measure >= threshold` and minimality; `split` gives
  `measure == threshold` (to 1e-9 relative). Volume is conserved across sealed and open bars.
* No lookahead: bars built from any prefix of a stream are exactly the first bars of the full
  run. Adaptive thresholds equal an independent recomputation from prior days only, and
  changing a future day's volume never changes an earlier threshold.

## Alignment: per-asset event bars and cross-sectional factors

Event bars are per-asset clocks. Ranking "the latest bar" across assets would compare bars
from different times, so factorflow separates the two jobs:

1. **Time-series factors run per asset on event bars.** `EventBars.panel(asset)` is a one-asset
   panel (rows are event bars, labeled by `end`) for `BatchEngine`. `EventPipeline(pipe,
   builder, history)` runs the same pipeline live: each sealed event bar seals a row, and the
   open event bar is the open row. That gives the same `is_open` revise/seal semantics and
   the same batch == incremental guarantee as clock bars.
2. **Cross-sectional factors run on a common clock grid**, sampled **as of the last sealed event
   bar**. `sample_panel(per_asset, grid_ends)` (batch) and `AsofSampler` (live) take, for each
   clock bar, the last event row with `end < grid_end`. The comparison is strict because a bar
   sealed by a trade at `ts` belongs to the clock bar containing `ts`. The result is an
   ordinary clock-grid `Panel`, so `rank`, `zscore` and `demean` (with `groupby` and masks) apply
   unchanged. Open event bars are never sampled, so cross-sectional values move only when an
   event bar seals.

## Performance notes

* Event-bar construction is a per-trade Python state machine, roughly 1-2 µs per trade, so
  about a second per million trades. A vectorized fixed-threshold fast path (cumsum plus
  searchsorted) is a possible later optimization, and it would have to match this state
  machine exactly.
* Clock bars: `build_time_bars` uses the same per-trade loop; history is normally loaded as
  bars (`Panel.from_frame`), not rebuilt from trades.
