"""The core guarantees, as properties over random trade streams:

1. every sealed live row == the batch row (bit-for-bit, NaN == NaN);
2. every mid-bar revision == batch run on (sealed history + the open bar treated as final);
on minute (24/7 and XNYS) and daily (XNYS) grids, with gaps (forward-filled bars).
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import factorflow as ff
from tests.conftest import assert_rows_equal, full_pipeline
from tests.helpers import random_trades, sectors

GRIDS = {
    "crypto-1min": (ff.BarGrid("24/7", "1min"), "2026-03-01", "2026-03-01"),
    "xnys-5min": (ff.BarGrid("XNYS", "5min"), "2026-11-25", "2026-11-30"),  # spans an early close
    "xnys-1d": (ff.BarGrid("XNYS", "1d"), "2026-01-02", "2026-03-31"),
}
PIPE = full_pipeline()
GRAPH = ff.compile_pipeline(PIPE)


def _panel(grid, d0, d1, trades, assets):
    s, e = grid.bars(d0, d1)
    p = ff.build_time_bars(trades, assets, s, e)
    p.columns["sector"] = np.tile(sectors(len(assets)), (len(e), 1))
    return p


def run_property(
    grid_name: str, seed: int, n: int, k: int, hist_frac: float, revise_every: int, partial: float
) -> None:
    grid, d0, d1 = GRIDS[grid_name]
    rng = np.random.default_rng(seed)
    s, e = grid.bars(d0, d1)
    e = e[: min(len(e), 160)]
    s = s[: len(e)]
    assets = tuple(f"S{i}" for i in range(n))
    trades = random_trades(rng, n, int(s[0]) - 1, int(e[-1]), k)
    full = ff.build_time_bars(trades, assets, s, e)
    full.columns["sector"] = np.tile(sectors(n), (len(e), 1))
    batch = ff.BatchEngine().run(GRAPH, full)

    h = max(1, int(len(e) * hist_frac))
    sealed: list[ff.Snapshot] = []
    live = ff.LiveEngine(
        GRAPH,
        full.head(h),
        grid,
        observers=[lambda snap: sealed.append(snap) if not snap.is_open else None],
        partial_threshold=partial,
    )
    assert live.open_end == e[h]
    start = np.searchsorted(trades.ts, e[h - 1], side="left")
    for i in range(start, len(trades)):
        live.tick(int(trades.asset[i]), int(trades.ts[i]), float(trades.price[i]), float(trades.size[i]))
        if (i - start) % revise_every == 0:
            snap = live.revise()
            assert snap.is_open
            row = {c: live.bar.row(c) for c in (*ff.bars.FIELDS, "has_trade")}
            row["sector"] = sectors(n)
            n_sealed = h + len(sealed)
            ref = ff.BatchEngine().run(GRAPH, full.head(n_sealed).with_row(row, live.open_end))
            assert_rows_equal(snap.values, ref.values, n_sealed, "open")
            if ref.screen is not None:
                assert np.array_equal(snap.screen, ref.screen[n_sealed])
    live.advance(int(e[-1]) + grid.grace_ns)
    assert len(sealed) == len(e) - h
    for j, snap in enumerate(sealed):
        assert snap.ts == e[h + j]
        assert_rows_equal(snap.values, batch.values, h + j, "sealed")


@given(
    seed=st.integers(0, 2**31),
    n=st.integers(1, 7),
    k=st.integers(1, 1500),
    hist=st.floats(0.05, 0.9),
    every=st.integers(1, 40),
    partial=st.sampled_from([0.0, 0.5, 1.1]),
)
def test_crypto_minute(seed, n, k, hist, every, partial):
    run_property("crypto-1min", seed, n, k, hist, every, partial)


@given(
    seed=st.integers(0, 2**31),
    n=st.integers(1, 7),
    k=st.integers(1, 1500),
    hist=st.floats(0.05, 0.9),
    every=st.integers(1, 40),
    partial=st.sampled_from([0.0, 0.5, 1.1]),
)
def test_xnys_intraday(seed, n, k, hist, every, partial):
    run_property("xnys-5min", seed, n, k, hist, every, partial)


@given(
    seed=st.integers(0, 2**31),
    n=st.integers(1, 7),
    k=st.integers(1, 1500),
    hist=st.floats(0.05, 0.9),
    every=st.integers(1, 40),
    partial=st.sampled_from([0.0, 0.5, 1.1]),
)
def test_xnys_daily(seed, n, k, hist, every, partial):
    run_property("xnys-1d", seed, n, k, hist, every, partial)


def test_revise_is_pure():
    """Revising twice, or revising then sealing, never changes sealed state."""
    grid = ff.BarGrid("24/7", "1min")
    s, e = grid.bars("2026-03-01", "2026-03-01")
    rng = np.random.default_rng(1)
    assets = ("A", "B", "C")
    tr = random_trades(rng, 3, int(s[0]), int(e[60]), 600)
    p = ff.build_time_bars(tr, assets, s[:60], e[:60])
    p.columns["sector"] = np.tile(sectors(3), (60, 1))
    live = ff.LiveEngine(GRAPH, p.head(30), grid)
    before = {t: b.data.copy() for t, b in live.bufs.items()}
    pos = {t: b.pos for t, b in live.bufs.items()}
    live.tick("A", int(e[29]) + 1, 101.0, 10)
    a = live.revise()
    b = live.revise()
    for k in a.values:
        assert np.array_equal(a.values[k], b.values[k], equal_nan=True)
    assert not b.changed
    for t, buf in live.bufs.items():
        assert buf.pos == pos[t]
        assert np.array_equal(buf.data[: buf.pos], before[t][: buf.pos], equal_nan=True)
