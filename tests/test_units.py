import numpy as np
import polars as pl
import pytest

import factorflow as ff
from factorflow import kernels as K

NS = 10**9


def ref_roll(x, w, fn):
    out = np.full(x.shape[:2], np.nan)
    for t in range(w - 1, x.shape[0]):
        out[t] = fn(x[t - w + 1 : t + 1])
    return out


@pytest.mark.parametrize("w", [2, 3, 7])
def test_rolling_kernels_match_reference(w):
    rng = np.random.default_rng(w)
    x = rng.standard_normal((40, 5))
    y = rng.standard_normal((40, 5))
    x[10, 2] = np.nan
    p = ff.Panel(np.arange(40), tuple("abcde"), {"x": x, "y": y})
    X, Y = ff.Column("x"), ff.Column("y")
    res = (
        ff.BatchEngine()
        .run(
            ff.Pipeline(
                {
                    "sum": X.rolling_sum(w),
                    "mean": X.rolling_mean(w),
                    "std": X.rolling_std(w),
                    "var": X.rolling_var(w),
                    "min": X.rolling_min(w),
                    "max": X.rolling_max(w),
                    "tsr": X.ts_rank(w),
                    "cov": X.rolling_cov(Y, w),
                    "corr": X.rolling_corr(Y, w),
                    "shift": X.shift(w),
                    "beta": X.rolling_beta(Y, w),
                }
            ),
            p,
        )
        .values
    )

    def cov(a, b):
        return ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / (len(a) - 1)

    np.testing.assert_allclose(res["sum"], ref_roll(x, w, lambda a: a.sum(0)), rtol=1e-12)
    np.testing.assert_allclose(res["mean"], ref_roll(x, w, lambda a: a.mean(0)), rtol=1e-12)
    np.testing.assert_allclose(res["std"], ref_roll(x, w, lambda a: a.std(0, ddof=1)), rtol=1e-10)
    np.testing.assert_allclose(res["var"], ref_roll(x, w, lambda a: a.var(0, ddof=1)), rtol=1e-10)
    np.testing.assert_allclose(res["min"], ref_roll(x, w, lambda a: a.min(0)))
    np.testing.assert_allclose(res["max"], ref_roll(x, w, lambda a: a.max(0)))
    np.testing.assert_allclose(
        res["tsr"], ref_roll(x, w, lambda a: np.where(np.isnan(a).any(0), np.nan, (a <= a[-1]).mean(0)))
    )
    xy = np.stack([x, y], -1)
    np.testing.assert_allclose(res["cov"], ref_roll(xy, w, lambda a: cov(a[..., 0], a[..., 1])), rtol=1e-9)
    np.testing.assert_allclose(
        res["corr"],
        ref_roll(
            xy, w, lambda a: cov(a[..., 0], a[..., 1]) / a[..., 0].std(0, ddof=1) / a[..., 1].std(0, ddof=1)
        ),
        rtol=1e-9,
    )
    sh = np.full_like(x, np.nan)
    sh[w:] = x[:-w]
    np.testing.assert_array_equal(res["shift"], sh)
    assert np.isnan(res["sum"][10 : 10 + w, 2]).all()


def test_ewm():
    x = np.array([[1.0], [np.nan], [3.0], [5.0]])
    out = (
        ff.BatchEngine()
        .run(ff.Pipeline({"e": ff.Column("x").ewm_mean(1.0)}), ff.Panel(np.arange(4), ("a",), {"x": x}))
        .values["e"][:, 0]
    )
    np.testing.assert_allclose(out, [1.0, 1.0, 2.0, 3.5])


def test_cross_sectional():
    v = np.array([3.0, 1.0, np.nan, 2.0, 2.0, 5.0])
    g = np.array([0, 0, 0, 1, 1, -1])
    m = np.array([True, True, True, True, False, True])
    np.testing.assert_array_equal(K.cs_rank(v, None, None), [4, 1, np.nan, 2, 3, 5.0])
    np.testing.assert_array_equal(K.cs_rank(v, m, g), [2, 1, np.nan, 1, np.nan, np.nan])
    np.testing.assert_array_equal(K.cs_rank(v, None, g, ascending=False), [1, 2, np.nan, 1, 2, np.nan])
    z = K.cs_zscore(v, None, g)
    np.testing.assert_allclose(z[:2], [1 / np.sqrt(2), -1 / np.sqrt(2)])
    assert np.isnan(z[3:]).all()  # group 1 has zero dispersion; -1 is missing
    np.testing.assert_allclose(K.cs_demean(v, m, None), [0.25, -1.75, np.nan, -0.75, np.nan, 2.25])


def test_cse_dedupes_structurally_equal_terms():
    c = ff.BarData.close
    g = ff.compile_pipeline(ff.Pipeline({"a": c.rolling_mean(5) + 1, "b": c.rolling_mean(5) * 2}))
    rolls = [t for t in g.order if type(t).__name__ == "Rolling"]
    assert len(rolls) == 1


def test_grid_xnys_minutes_and_early_close():
    g = ff.BarGrid("XNYS", "1min")
    s, e = g.bars("2026-11-27", "2026-11-27")  # day after Thanksgiving, 13:00 close
    assert len(e) == 210
    s, e = g.bars("2026-11-30", "2026-11-30")
    assert len(e) == 390
    assert e[0] == ff.to_ns("2026-11-30T14:31:00+00:00") and s[0] == ff.to_ns("2026-11-30T14:30:00+00:00")
    assert len(g.bars("2026-11-26", "2026-11-26")[0]) == 0  # Thanksgiving
    d = ff.BarGrid("XNYS", "1d")
    s, e = d.bars("2026-11-23", "2026-11-27")
    assert len(e) == 4 and d.grace_ns == 0 and g.grace_ns == 2 * NS
    s, e = ff.BarGrid("24/7", "1h").bars("2026-03-01", "2026-03-01")
    assert len(e) == 24
    _s2, e2 = g.bars_after(int(ff.to_ns("2026-11-27T18:00:00+00:00")), 2)
    assert e2[0] == ff.to_ns("2026-11-30T14:31:00+00:00")


def _crypto_engine(pipe, n_hist=5):
    grid = ff.BarGrid("24/7", "1min")
    s, e = grid.bars("2026-03-01", "2026-03-01")
    tr = ff.Trades([0, 1], [int(s[0]) + 1, int(s[0]) + 2], [10.0, 20.0], [1.0, 1.0])
    hist = ff.build_time_bars(tr, ("A", "B"), s[:n_hist], e[:n_hist])
    return grid, e, hist, ff.LiveEngine(pipe, hist, grid)


def test_missing_bars_forward_fill_and_open_flag():
    c, v = ff.BarData.close, ff.BarData.volume
    grid, e, hist, live = _crypto_engine(
        ff.Pipeline({"c": c, "v": v, "ht": ff.where(ff.BarData.has_trade, 1, 0)})
    )
    assert np.array_equal(hist.columns["close"][-1], [10.0, 20.0])
    assert np.array_equal(hist.columns["volume"][-1], [0.0, 0.0])
    snaps = live.advance(int(e[7]) + grid.grace_ns)  # three empty bars
    assert [s.ts for s in snaps] == list(e[5:8])
    assert all(not s.is_open for s in snaps)
    assert np.array_equal(snaps[-1].values["c"], [10.0, 20.0])
    assert np.array_equal(snaps[-1].values["v"], [0.0, 0.0])
    live.tick("A", int(e[7]) + 5, 11.0, 3.0)
    snap = live.revise()
    assert snap.is_open and snap.ts == e[8]
    assert snap.values["c"].tolist() == [11.0, 20.0] and snap.values["ht"].tolist() == [1.0, 0.0]
    assert snap.changed == {"c", "v", "ht"}
    assert "is_open" in snap.frame().columns


def test_grace_period_routing_and_late_ticks():
    c = ff.BarData.close
    _grid, e, _hist, live = _crypto_engine(ff.Pipeline({"c": c}))
    end = int(e[5])
    live.tick("A", end - 1, 12.0, 1.0)  # in the open bar
    live.tick("A", end + NS, 13.0, 1.0)  # next bar, but inside grace: held
    assert live.open_end == end and live.revise().values["c"][0] == 12.0
    live.tick("B", end - 10, 21.0, 1.0)  # late print within grace: still accepted
    assert live.revise().values["c"][1] == 21.0
    live.tick("B", end + 3 * NS, 22.0, 1.0)  # past grace -> seals, then belongs to the new bar
    assert live.open_end == int(e[6]) and live.last_sealed_ts == end
    assert live.revise().values["c"].tolist() == [13.0, 22.0]
    live.tick("A", end - 1, 99.0, 1.0)  # bar already sealed: dropped
    assert live.late_ticks == 1


def test_cutoff_and_observers():
    c = ff.BarData.close
    seen = []
    pipe = ff.Pipeline({"up": ff.where(c > 15.0, 1.0, 0.0), "rk": c.rank()})
    _grid, e, _hist, live = _crypto_engine(pipe)
    live.observers.append(seen.append)
    live.revise()
    seen.clear()
    live.tick("A", int(e[4]) + 1, 10.0, 5.0)  # same close: nothing changes
    snap = live.revise()
    assert not snap.changed and not seen
    live.tick("A", int(e[4]) + 2, 16.0, 1.0)
    snap = live.revise()
    assert snap.changed == {"up"} and len(seen) == 1  # rank of A unchanged (still 1 of 2)
    live.tick("A", int(e[4]) + 3, 25.0, 1.0)
    assert live.revise().changed == {"rk"}


def test_set_bar_and_set_open():
    pipe = ff.Pipeline({"x": ff.Column("signal") * ff.BarData.close})
    grid = ff.BarGrid("24/7", "1min")
    _s, e = grid.bars("2026-03-01", "2026-03-01")
    hist = ff.Panel(e[:3], ("A", "B"), {"close": np.ones((3, 2)), "signal": np.full((3, 2), 2.0)})
    live = ff.LiveEngine(pipe, hist, grid)
    assert live.revise().values["x"].tolist() == [2.0, 2.0]
    live.set_bar("B", 1, 3, 1, 3, 100)
    live.set_open("signal", {"A": 5.0})
    assert live.revise().values["x"].tolist() == [5.0, 6.0]
    snap = live.seal()
    assert not snap.is_open and snap.ts == e[3]
    assert live.revise().values["x"].tolist() == [5.0, 6.0]  # carried forward


def test_panel_from_frame_and_frames():
    df = pl.DataFrame({"ts": [1, 1, 2], "asset": ["a", "b", "a"], "close": [1.0, 2.0, 3.0]})
    p = ff.Panel.from_frame(df)
    assert p.shape == (2, 2) and np.isnan(p.columns["close"][1, 1])
    res = ff.BatchEngine().run(ff.Pipeline({"c": ff.BarData.close}, screen=ff.BarData.close.notnull()), p)
    f = res.frame(open_last=True)
    assert f.height == 3 and f["is_open"].to_list() == [False, False, True]


def test_validation_errors():
    c = ff.BarData.close
    with pytest.raises(ValueError):
        c.rolling_std(1)
    with pytest.raises(ValueError):
        c.shift(0)
    with pytest.raises(KeyError):
        ff.BatchEngine().run(ff.Pipeline({"x": ff.Column("nope")}), ff.Panel([1], ("a",), {}))
    with pytest.raises(ValueError):
        ff.Panel([2, 1], ("a",), {})
    with pytest.raises(ValueError):
        ff.BarGrid("XNYS", "3fortnights")
