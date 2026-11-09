"""The simulated market, the replay driver and the validation suite (factorflow.sim)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np
import pytest

import factorflow as ff
from factorflow.grid import NS, BarGrid
from factorflow.sim import STYLES, SimConfig, simulate
from factorflow.sim.pipelines import crypto_pipeline, daily_pipeline, event_pipeline, minute_pipeline
from factorflow.sim.replay import replay, verify_event_pipeline, verify_replay
from factorflow.sim.validate import bar_distributions, decile_backtest, factor_recovery, moments, signal_ic


@pytest.fixture(scope="module")
def small():
    return simulate(SimConfig(seed=3, history_days=30, intraday_days=2).scaled(n=12))


@pytest.fixture(scope="module")
def medium():
    return simulate(SimConfig(seed=11, history_days=252, intraday_days=1).scaled(n=150))


# ---------------------------------------------------------------- generator
def test_deterministic_for_a_seed(small):
    again = simulate(SimConfig(seed=3, history_days=30, intraday_days=2).scaled(n=12))
    for k, v in small.equities.daily.columns.items():
        np.testing.assert_array_equal(v, again.equities.daily.columns[k])
    for a, b in ((small.equities.trades, again.equities.trades), (small.crypto.trades, again.crypto.trades)):
        np.testing.assert_array_equal(a.ts, b.ts)
        np.testing.assert_array_equal(a.price, b.price)
        np.testing.assert_array_equal(a.size, b.size)
    other = simulate(SimConfig(seed=4, history_days=30, intraday_days=2).scaled(n=12))
    assert not np.array_equal(other.equities.daily.columns["close"], small.equities.daily.columns["close"])


def test_trades_are_sorted_and_inside_sessions(small):
    for sim in (small.equities, small.crypto):
        t = sim.trades
        assert np.all(np.diff(t.ts) >= 0)
        assert np.all(t.price > 0) and np.all(t.size > 0)
        day = np.searchsorted(sim.closes, t.ts, side="left")
        assert np.all(t.ts >= sim.opens[day]) and np.all(t.ts <= sim.closes[day])
        assert set(np.unique(day)) <= set(sim.intraday.tolist())


def test_early_close_and_halts(small):
    eq = small.equities
    # 2026-11-27 (day after Thanksgiving) closes at 13:00 ET = 18:00 UTC
    i = int(np.flatnonzero(eq.sessions == np.datetime64("2026-11-27"))[0])
    assert eq.closes[i] == int(datetime(2026, 11, 27, 18, tzinfo=UTC).timestamp()) * NS
    for a, h0, h1 in eq.halts:
        m = (eq.trades.asset == a) & (eq.trades.ts > h0) & (eq.trades.ts < h1)
        assert not m.any()


def test_crypto_trades_on_weekends(small):
    cr = small.crypto
    days = cr.trades.ts.astype("datetime64[ns]").astype("datetime64[D]")
    wd = (days.astype(np.int64) + 3) % 7  # Monday = 0
    assert cr.calendar == "24/7"
    assert (wd >= 5).any()


def test_intraday_daily_rows_come_from_the_ticks(small):
    for sim in (small.equities, small.crypto):
        c = sim.daily.columns
        for r in sim.intraday:
            m = np.searchsorted(sim.closes, sim.trades.ts, side="left") == r
            for j in range(len(sim.assets)):
                mj = m & (sim.trades.asset == j)
                if mj.any():
                    assert c["close"][r, j] == sim.trades.price[mj][-1]
                    assert c["volume"][r, j] == pytest.approx(sim.trades.size[mj].sum())


def test_truth_is_consistent(small):
    tr = small.equities.truth
    _, N = tr.log_returns.shape
    assert tr.beta.shape == (N,) and set(tr.exposures) == set(STYLES)
    assert tr.kappa == pytest.approx(tr.ic / np.sqrt(1 - tr.ic**2))
    s = tr.signal[np.isfinite(tr.signal).all(axis=1)]
    np.testing.assert_allclose(s.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(s.std(axis=1), 1, atol=1e-12)
    assert np.all(tr.total_vol > tr.idio_vol)


# ---------------------------------------------------------------- replay == batch
def test_replay_daily_bars_equal_batch(small):
    eq = small.equities
    v = verify_replay(
        daily_pipeline(len(eq.assets), beta_window=20),
        eq.daily,
        eq.trades,
        eq.daily_grid(),
        eq.n_hist,
        extra_columns=("signal", "industry", *STYLES),
        revise_every_ns=600 * NS,
        check_open_every=3,
    )
    assert v.ok, v.mismatches[:5]
    assert v.sealed_rows == 2 and v.open_checked > 0


@pytest.mark.parametrize("freq", ["1min", "5min"])
def test_replay_minute_bars_equal_batch(small, freq):
    eq = small.equities
    p, _ = eq.time_bars(freq)
    first = int(np.searchsorted(p.times, eq.closes[eq.intraday[0]], side="right"))
    v = verify_replay(
        minute_pipeline(),
        p,
        eq.trades,
        BarGrid("XNYS", freq),
        first,
        extra_columns=("signal", "industry"),
        revise_every_ticks=97,
        revise_every_ns=None,
        check_open_every=5,
    )
    assert v.ok, v.mismatches[:5]
    assert v.sealed_rows == len(p.times) - first  # includes the 13:00 ET early close


def test_replay_crypto_247_equal_batch_with_partial_threshold(small):
    cr = small.crypto
    p, _ = cr.time_bars("5min")
    first = int(np.searchsorted(p.times, cr.closes[cr.intraday[0]], side="right"))
    v = verify_replay(
        crypto_pipeline(),
        p,
        cr.trades,
        BarGrid("24/7", "5min"),
        first,
        revise_every_ns=30 * NS,
        check_open_every=7,
        partial_threshold=0.5,
    )
    assert v.ok, v.mismatches[:5]
    assert v.sealed_rows == 288


@pytest.mark.parametrize("kind", ["tick", "volume", "dollar"])
def test_event_pipeline_equals_batch(small, kind):
    cr = small.crypto
    m = cr.trades.asset == 1
    measure = {
        "tick": float(m.sum()),
        "volume": cr.trades.size[m].sum(),
        "dollar": (cr.trades.size[m] * cr.trades.price[m]).sum(),
    }[kind]
    thr = float(measure) / 400
    v = verify_event_pipeline(
        event_pipeline(), cr.trades, 1, lambda: ff.EventBarBuilder(kind, thr, overshoot="split"), 0.37
    )
    assert v.ok, v.mismatches[:5]
    assert v.sealed_rows > 100


def test_event_pipeline_resumed_mid_bar_shows_the_open_bar(small):
    """Regression: an EventPipeline built on a builder holding an open bar ignored that bar
    until the next trade, so an immediate revise showed a stale (flat) open row."""
    cr = small.crypto
    m = cr.trades.asset == 0
    ts, px, sz = cr.trades.ts[m][:2000], cr.trades.price[m][:2000], cr.trades.size[m][:2000]
    b = ff.EventBarBuilder("volume", float(sz.sum()) / 20)
    hist = ff.EventBars.from_bars(b.update_many(ts.tolist(), px.tolist(), sz.tolist()))
    assert b.open_bar is not None
    ep = ff.EventPipeline(event_pipeline(), b, history=hist)
    snap = ep.revise()
    assert snap.is_open
    assert snap.values["logc"][0] == pytest.approx(np.log(b.open_bar.close))


def test_replay_speed_mode_seals_on_the_clock(small):
    """Accelerated real time: bars with no trades still seal as the clock passes them."""
    eq = small.equities
    p, _ = eq.time_bars("1min")
    first = int(np.searchsorted(p.times, eq.closes[eq.intraday[0]], side="right"))
    live = ff.LiveEngine(minute_pipeline(), p.head(first), BarGrid("XNYS", "1min"))
    sealed = []
    start = int(np.searchsorted(eq.trades.ts, int(p.times[first - 1]), side="right"))
    stop = int(np.searchsorted(eq.trades.ts, int(p.times[first + 4]), side="right"))
    t0 = time.perf_counter()
    replay(live, eq.trades, start=start, stop=stop, speed=600.0, on_seal=sealed.append)
    assert time.perf_counter() - t0 < 5
    assert len(sealed) >= 4
    assert all(not s.is_open for s in sealed)


# ---------------------------------------------------------------- BarGrid cache
@pytest.mark.parametrize(
    ("cal", "freq"), [("XNYS", "1min"), ("XNYS", "1d"), ("24/7", "5min"), ("XNYS", "30min")]
)
def test_bargrid_cache_matches_fresh_grid(cal, freq):
    """Regression: bars_after rebuilt the calendar on every seal; it now caches a block."""
    g = BarGrid(cal, freq)
    rng = np.random.default_rng(0)
    lo = int(datetime(2026, 1, 2, tzinfo=UTC).timestamp()) * NS
    hi = int(datetime(2026, 12, 20, tzinfo=UTC).timestamp()) * NS
    pts = np.sort(rng.integers(lo, hi, 150))
    pts = np.concatenate([pts, pts[::-1][:30]])  # also walk backwards
    for t in pts.tolist():
        count = int(rng.integers(1, 4))
        s, e = g.bars_after(t, count)
        s0, e0 = BarGrid(cal, freq).bars_after(t, count)
        np.testing.assert_array_equal(s, s0)
        np.testing.assert_array_equal(e, e0)
        assert e[0] > t


def test_bargrid_bars_after_is_fast_when_cached():
    g = BarGrid("XNYS", "1min")
    _, e = g.bars("2026-11-02", "2026-11-06")
    g.bars_after(int(e[0]))
    t0 = time.perf_counter()
    for t in e[:1000].tolist():
        g.bars_after(t)
    assert (time.perf_counter() - t0) / 1000 < 50e-6  # was ~330 us per call uncached


# ---------------------------------------------------------------- validation
def test_moments_of_a_normal_sample():
    x = np.random.default_rng(1).standard_normal(200_000)
    m = moments(x)
    assert abs(m["skew"]) < 0.02 and abs(m["exkurt"]) < 0.05 and m["jb_p"] > 1e-3


def test_factor_and_signal_recovery(medium):
    eq = medium.equities
    rec = factor_recovery(eq)
    for k in ("market", *STYLES):
        assert rec.factors[k]["corr"] > 0.7, (k, rec.factors[k])
        assert 0.85 < rec.factors[k]["slope"] < 1.15, (k, rec.factors[k])
    assert rec.beta["corr"] > 0.8
    ic = signal_ic(eq, rec=rec)
    assert abs(ic.z) < 3.5, ic
    assert ic.resid_ic > 0 and ic.raw_ic > 0
    bt = decile_backtest(eq)
    assert bt.mean_bp > 0 and bt.tstat > 1.5


def test_event_bars_are_more_normal_than_clock_bars(small):
    d = bar_distributions(small.equities)
    assert d["volume"]["exkurt"] < d["5min"]["exkurt"] < d["1min"]["exkurt"]
    assert d["dollar"]["exkurt"] < d["5min"]["exkurt"]
    assert d["volume"]["jb"] < d["5min"]["jb"]


@pytest.mark.slow
def test_cli_demo_runs(capsys):
    from factorflow.sim.__main__ import main

    assert (
        main(
            [
                "demo",
                "--names",
                "20",
                "--history-days",
                "60",
                "--intraday-days",
                "2",
                "--seconds",
                "1",
                "--speed",
                "600",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "OK" in out and "FAIL" not in out
