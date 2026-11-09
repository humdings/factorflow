"""Self-contained factorflow quickstart on synthetic data.

No market feed and no saved fixtures: prices are a seeded random walk, and the
clock comes from ``BarGrid``. Run from a checkout with the package installed:

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

import factorflow as ff

NS = 1_000_000_000


class RangePos(ff.Factor):
    """Where the latest value sits in its rolling high-low range, in ``[0, 1]``.

    A window with any missing input, or a zero range, is missing (NaN).
    ``compute`` receives ``(T_in, N)`` blocks and returns ``T_in - window + 1`` rows.
    Batch calls it on the whole history. Live calls it on exactly ``window`` rows
    and reads row 0, so both paths run this function.
    """

    def __init__(self, source: ff.Factor, window: int):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.inputs = (source,)
        self.window = int(window)
        self.params = (self.window,)

    def compute(self, blocks, prev=None):
        x = np.asarray(blocks[0], dtype=np.float64)
        w = self.window
        out = np.empty((x.shape[0] - w + 1, x.shape[1]), dtype=np.float64)
        for i in range(out.shape[0]):
            window = x[i : i + w]
            lo = np.min(window, axis=0)
            hi = np.max(window, axis=0)
            span = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                pos = (window[-1] - lo) / span
            missing = np.isnan(window).any(axis=0) | (span == 0)
            out[i] = np.where(missing, np.nan, pos)
        return out


def main() -> None:
    rng = np.random.default_rng(0)
    assets = ("AAA", "BBB", "CCC", "DDD")
    n_bars, n = 40, len(assets)

    # 24/7 one-minute bars. grace=0 seals a bar as soon as time reaches its end.
    # "5min", "1h" and "1d" are the same idea; "XNYS" uses the exchange calendar.
    grid = ff.BarGrid("24/7", "1min", grace=0)
    starts, ends = grid.bars("2026-01-05", "2026-01-05")
    starts, ends = starts[:n_bars], ends[:n_bars]

    shock = rng.normal(0, 0.004, size=(n_bars, n))
    close = 100 * np.exp(np.cumsum(shock, axis=0))
    volume = rng.integers(50, 500, size=(n_bars, n)).astype(np.float64)
    sector = np.tile(np.array([0, 0, 1, 1], dtype=np.int64), (n_bars, 1))
    panel = ff.Panel(ends, assets, {"close": close, "volume": volume, "sector": sector}, starts)

    close_t = ff.Column("close")
    volume_t = ff.Column("volume")
    sector_t = ff.Column("sector", ff.Classifier)
    ret = close_t.pct_change(1)
    universe = (close_t * volume_t).rolling_mean(10).top(3)
    score = close_t.pct_change(10).zscore(groupby=sector_t, mask=universe)
    pipe = ff.Pipeline({"score": score, "ret": ret, "range_pos": RangePos(close_t, 10)}, screen=universe)

    batch = ff.BatchEngine().run(pipe, panel)
    print(batch.frame().tail(4))

    # Warm the live engine on every sealed bar but the last, then revise that open bar.
    live = ff.LiveEngine(pipe, panel.head(n_bars - 1), grid)
    last = n_bars - 1
    live.set_open("close", close[last])
    live.set_open("volume", volume[last])
    live.set_open("sector", sector[last])
    snap = live.revise()
    assert snap.is_open
    sealed = live.seal()
    assert sealed.is_open is False
    for name in ("score", "ret", "range_pos"):
        np.testing.assert_array_equal(sealed.values[name], batch.values[name][last])
    print("sealed live row matches batch, including the custom factor")

    # Information-driven bars. Twelve trades of 10 shares close a bar every 30 shares.
    ts = np.arange(12) * NS
    price = 100 + np.arange(12, dtype=np.float64)
    size = np.full(12, 10.0)
    bars, open_bar = ff.build_event_bars(ts, price, size, "volume", ff.Fixed(30))
    assert len(bars) == 4 and open_bar is None
    print(f"volume bars: {len(bars)} sealed, no open bar")

    # Clock bars from a few trades. Empty minutes carry the previous close forward.
    base = int(starts[0])
    trade_ts = np.array([base + NS, base + 20 * NS, base + 5 * NS])
    order = np.argsort(trade_ts)
    trades = ff.Trades(
        asset=np.array([0, 0, 1])[order],
        ts=trade_ts[order],
        price=np.array([10.0, 12.0, 20.0])[order],
        size=np.array([5.0, 1.0, 4.0])[order],
    )
    built = ff.build_time_bars(trades, ("AAA", "BBB"), starts[:3], ends[:3])
    np.testing.assert_array_equal(built.columns["close"][:, 0], [12.0, 12.0, 12.0])
    np.testing.assert_array_equal(built.columns["volume"][:, 0], [6.0, 0.0, 0.0])
    print("clock bars: first minute traded, later minutes forward-filled the close")

    xnys = ff.BarGrid("XNYS", "1min")
    early = xnys.bars("2026-11-27", "2026-11-27")[1]
    regular = xnys.bars("2026-11-30", "2026-11-30")[1]
    holiday = xnys.bars("2026-11-26", "2026-11-26")[1]
    print(f"XNYS 1min bars: early close {len(early)}, regular session {len(regular)}, holiday {len(holiday)}")


if __name__ == "__main__":
    main()
