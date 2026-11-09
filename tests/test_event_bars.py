"""Event (tick / volume / dollar) bars: batch == incremental, thresholds respected, no lookahead."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import factorflow as ff
from factorflow.events import utc_day
from tests.conftest import assert_rows_equal

DAY = 86400 * 10**9


@st.composite
def trade_stream(draw, max_n=400, days=6):
    n = draw(st.integers(1, max_n))
    seed = draw(st.integers(0, 2**31))
    rng = np.random.default_rng(seed)
    t0 = 1_767_225_600 * 10**9  # 2026-01-01
    ts = np.sort(rng.integers(t0, t0 + days * DAY, n))
    if draw(st.booleans()):  # duplicate timestamps
        ts = np.sort(np.repeat(ts[: (n + 1) // 2], 2)[:n])
    price = np.round(50 + rng.random(n) * 50, 2)
    heavy = draw(st.booleans())
    size = (rng.pareto(1.2, n) * 50 + 1 if heavy else rng.integers(1, 300, n)).astype(float)
    return ts, price, size


rules = st.one_of(
    st.builds(ff.Fixed, st.floats(1.0, 20_000.0)),
    st.builds(
        ff.AdaptiveDaily, st.floats(1.0, 50.0), st.integers(1, 5), st.floats(10.0, 5000.0), st.integers(1, 3)
    ),
)
kinds = st.sampled_from(["tick", "volume", "dollar"])
policies = st.sampled_from(["close", "split"])


def _measure(kind, price, size):
    return {"tick": np.ones_like(size), "volume": size, "dollar": price * size}[kind]


def _rule_for(kind, rule):
    if kind == "tick" and isinstance(rule, ff.Fixed):
        return ff.Fixed(max(1.0, round(rule.value / 1000)))
    if kind == "dollar" and isinstance(rule, ff.Fixed):
        return ff.Fixed(rule.value * 50)
    return rule


@given(trade_stream(), kinds, rules, policies, st.lists(st.integers(1, 50), min_size=1, max_size=20))
def test_batch_equals_incremental(stream, kind, rule, policy, chunks):
    ts, price, size = stream
    rule = _rule_for(kind, rule)
    bars, open_bar = ff.build_event_bars(ts, price, size, kind, rule, policy)
    b = ff.EventBarBuilder(kind, rule, policy)
    got, i, c = [], 0, 0
    while i < len(ts):
        step = chunks[c % len(chunks)]
        got += b.update_many(ts[i : i + step], price[i : i + step], size[i : i + step])
        i += step
        c += 1
    inc = ff.EventBars.from_bars(got)
    assert len(inc) == len(bars)
    for k in ff.events.EVENT_FIELDS:
        assert np.array_equal(inc[k], bars[k]), k
    ob = b.open_bar
    assert (ob is None) == (open_bar is None)
    if ob is not None:
        assert ob.as_tuple() == open_bar.as_tuple()


@given(trade_stream(), kinds, rules, policies)
def test_thresholds_and_conservation(stream, kind, rule, policy):
    ts, price, size = stream
    rule = _rule_for(kind, rule)
    bars, ob = ff.build_event_bars(ts, price, size, kind, rule, policy)
    m = _measure(kind, bars["close"], bars["volume"])
    if kind == "dollar":
        m = bars["dollar"]
    elif kind == "tick":
        m = bars["n_ticks"].astype(float)
    thr = bars["threshold"]
    eff = "close" if kind == "tick" else policy
    if len(bars):
        if eff == "close":
            assert np.all(m >= thr * (1 - 1e-12))
        else:
            np.testing.assert_allclose(m, thr, rtol=1e-9)
        assert np.all(bars["start"] <= bars["end"])
        assert np.all(np.diff(bars["end"]) >= 0)
        assert np.all(bars["low"] <= bars["high"])
    # every unit of volume lands in exactly one bar
    tot = bars["volume"].sum() + (ob.volume if ob else 0.0)
    np.testing.assert_allclose(tot, size.sum(), rtol=1e-9)
    if eff == "close" and len(bars):
        # minimality: dropping the crossing trade leaves the bar below threshold
        ends = np.cumsum(bars["n_ticks"]) - 1  # whole trades, in order
        last = _measure(kind, price, size)[ends]
        # (the threshold may change at a day boundary while a bar is open, so only check
        # bars that lived within one day)
        same_day = (bars["start"] // DAY) == (bars["end"] // DAY)
        assert np.all((m - last < thr * (1 + 1e-12))[same_day])


@given(trade_stream(), kinds, rules, policies, st.floats(0.05, 0.95))
def test_no_lookahead_prefix(stream, kind, rule, policy, frac):
    """Bars built from a prefix of the stream are exactly the full run's first bars."""
    ts, price, size = stream
    rule = _rule_for(kind, rule)
    full, _ = ff.build_event_bars(ts, price, size, kind, rule, policy)
    k = max(1, int(len(ts) * frac))
    pre, _ = ff.build_event_bars(ts[:k], price[:k], size[:k], kind, rule, policy)
    for f in ff.events.EVENT_FIELDS:
        assert np.array_equal(pre[f], full[f][: len(pre)]), f


@given(trade_stream(days=10), st.sampled_from(["volume", "dollar"]), st.integers(1, 4), st.floats(1, 20))
def test_adaptive_threshold_point_in_time(stream, kind, lookback, per_day):
    """The threshold used on day d is the mean measure of the previous `lookback` traded days / target."""
    ts, price, size = stream
    rule = ff.AdaptiveDaily(per_day, lookback, initial=1234.0)
    b = ff.EventBarBuilder(kind, rule, "close")
    meas = _measure(kind, price, size)
    days = ts // DAY
    uniq = np.unique(days)
    totals = {d: meas[days == d].sum() for d in uniq}
    for t, p, q in zip(ts, price, size, strict=True):
        b.update(t, p, q)
        d = t // DAY
        prior = [totals[x] for x in uniq if x < d][-lookback:]
        expect = 1234.0 if not prior else np.mean(prior) / per_day
        assert b.threshold == pytest.approx(expect, rel=1e-9)
    # perturbing a future day never changes an earlier threshold
    if len(uniq) > 1:
        cut = uniq[-1]
        sz2 = size.copy()
        sz2[days == cut] *= 7
        a, _ = ff.build_event_bars(ts, price, size, kind, rule, "close")
        c, _ = ff.build_event_bars(ts, price, sz2, kind, rule, "close")
        early_a = a["end"] < cut * DAY
        early_c = c["end"] < cut * DAY
        assert np.array_equal(a["threshold"][early_a], c["threshold"][early_c])


def test_split_large_trade_fills_several_bars():
    bars, ob = ff.build_event_bars([1, 2], [10.0, 10.0], [5.0, 23.0], "volume", 10.0, "split")
    assert list(bars["volume"]) == [10.0, 10.0]
    assert list(bars["end"]) == [2, 2]
    assert ob is not None and ob.volume == pytest.approx(8.0)
    bars2, ob2 = ff.build_event_bars([1, 2], [10.0, 10.0], [5.0, 23.0], "volume", 10.0, "close")
    assert list(bars2["volume"]) == [28.0] and ob2 is None


def test_tick_bars():
    bars, ob = ff.build_event_bars(np.arange(10), np.arange(10.0) + 1, np.ones(10), "tick", 3)
    assert list(bars["n_ticks"]) == [3, 3, 3] and list(bars["close"]) == [3.0, 6.0, 9.0]
    assert ob is not None and ob.n_ticks == 1


def test_validation():
    with pytest.raises(ValueError):
        ff.EventBarBuilder("imbalance")
    with pytest.raises(ValueError):
        ff.EventBarBuilder("volume", ff.AdaptiveDaily(10))
    with pytest.raises(ValueError):
        ff.Fixed(0)


# ---------------------------------------------------------------- per-asset pipelines
EV_PIPE = ff.Pipeline(
    {
        "ret": ff.BarData.close.pct_change(1),
        "vol": ff.BarData.close.pct_change(1).rolling_std(4),
        "ewm": ff.BarData.close.ewm_mean(2.0),
        "dur": ff.BarData.n_ticks.rolling_mean(3),
    }
)


@given(trade_stream(max_n=300), kinds, policies, st.floats(0.0, 0.9), st.integers(1, 15))
def test_event_pipeline_live_equals_batch(stream, kind, policy, frac, every):
    ts, price, size = stream
    rule = {"tick": ff.Fixed(5), "volume": ff.Fixed(800.0), "dollar": ff.Fixed(60_000.0)}[kind]
    k = int(len(ts) * frac)
    b = ff.EventBarBuilder(kind, rule, policy)
    hist = ff.EventBars.from_bars(b.update_many(ts[:k], price[:k], size[:k]))
    ep = ff.EventPipeline(EV_PIPE, b, hist)
    full, _ = ff.build_event_bars(ts, price, size, kind, rule, policy)
    batch = ff.BatchEngine().run(EV_PIPE, full.panel())
    sealed = len(hist)
    for i in range(k, len(ts)):
        for snap in ep.update(ts[i], price[i], size[i]):
            if not snap.is_open:
                assert_rows_equal(snap.values, batch.values, sealed, "sealed")
                sealed += 1
            elif (i - k) % every == 0:
                ob = b.open_bar
                so_far = ff.EventBars.from_bars(
                    [ff.EventBar(*[full[f][r] for f in ff.events.EVENT_FIELDS]) for r in range(sealed)] + [ob]
                )
                ref = ff.BatchEngine().run(EV_PIPE, so_far.panel())
                assert_rows_equal(snap.values, ref.values, sealed, "open")
    assert sealed == len(full)


# ---------------------------------------------------------------- clock alignment
@given(st.integers(0, 2**31), st.integers(1, 5))
def test_asof_sampler_matches_batch_and_has_no_lookahead(seed, n):
    rng = np.random.default_rng(seed)
    grid = ff.BarGrid("24/7", "1min")
    s, e = grid.bars("2026-03-01", "2026-03-01")
    e = e[:120]
    assets = tuple(f"C{i}" for i in range(n))
    per = {}
    sampler = ff.AsofSampler(assets, ("close", "ret"))
    events = []
    for a in assets:
        m = int(rng.integers(0, 200))
        ends = np.sort(rng.integers(int(s[0]), int(e[-1]) + 10**9, m))
        close = rng.random(m)
        ret = rng.standard_normal(m)
        per[a] = (ends, {"close": close, "ret": ret})
        events += [(int(t), a, {"close": c, "ret": r}) for t, c, r in zip(ends, close, ret, strict=True)]
    batch = ff.sample_panel(per, e, assets)
    events.sort(key=lambda x: x[0])
    i = 0
    for r, ge in enumerate(e):
        # in live, event bars that seal during the clock bar's grace period arrive before it seals
        while i < len(events) and events[i][0] < ge + grid.grace_ns:
            sampler.push(events[i][1], events[i][0], events[i][2])
            i += 1
        row = sampler.row(int(ge))
        for k in ("close", "ret"):
            assert np.array_equal(row[k], batch.columns[k][r], equal_nan=True)
        for j, a in enumerate(assets):
            ends, vals = per[a]
            ok = ends < ge
            exp = vals["close"][ok][-1] if ok.any() else np.nan
            assert np.array_equal(batch.columns["close"][r, j], exp, equal_nan=True)


def test_cross_sectional_rank_on_sampled_event_bars():
    rng = np.random.default_rng(3)
    grid = ff.BarGrid("24/7", "5min")
    s, e = grid.bars("2026-03-01", "2026-03-01")
    per = {}
    for a in ("BTC", "ETH", "SOL"):
        m = 3000
        ts = np.sort(rng.integers(int(s[0]), int(e[-1]), m))
        px = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, m)))
        bars, _ = ff.build_event_bars(
            ts, px, rng.integers(1, 10, m).astype(float), "dollar", ff.Fixed(20_000.0)
        )
        res = ff.BatchEngine().run(EV_PIPE, bars.panel(a))
        per[a] = (
            bars["end"],
            {"close": bars["close"], "ret": res.values["ret"][:, 0], "vol": res.values["vol"][:, 0]},
        )
    clock = ff.sample_panel(per, e)
    pipe = ff.Pipeline({"rk": ff.Column("vol").rank(), "z": ff.Column("ret").zscore()})
    out = ff.BatchEngine().run(pipe, clock)
    assert np.nanmax(out.values["rk"]) == 3.0
    assert utc_day(int(e[0])) == utc_day(int(s[0]))
