"""Example pipelines used by the demo, the validation and the benchmarks."""

from __future__ import annotations

from factorflow.graph import Pipeline
from factorflow.sim.market import STYLES
from factorflow.terms import BarData, Classifier, Column, Factor, Filter, where

close, volume, dollar = BarData.close, BarData.volume, BarData.dollar
high, low = BarData.high, BarData.low
industry: Classifier = Column("industry", Classifier)
signal: Factor = Column("signal")


def beta_window_for(history_days: int) -> int:
    """200 days, or the history less a month when that is shorter (at least 20)."""
    return max(20, min(200, history_days - 30))


def daily_pipeline(n_assets: int, beta_window: int = 200) -> Pipeline:
    """Daily research pipeline: returns, beta to the equal-weight market, styles, the planted
    signal and decile screens."""
    r = close.pct_change(1)
    logret = (close / close.shift(1)).log()
    mkt = r - r.demean()  # equal-weight market return, broadcast to every asset
    universe: Filter = (dollar.rolling_mean(20) > 0) & close.notnull()
    decile = max(1, n_assets // 10)
    cols = {
        "ret": r,
        "logret": logret,
        "mkt": mkt,
        "beta": r.rolling_beta(mkt, beta_window),
        "vol20": r.rolling_std(20),
        "mom20": close.pct_change(20),
        "ind_rel": r.demean(groupby=industry),
        "signal_z": signal.zscore(mask=universe),
        "long": signal.top(decile, mask=universe),
        "short": signal.bottom(decile, mask=universe),
        **{f"{k}_z": Column(k).zscore(mask=universe) for k in STYLES},
    }
    return Pipeline(cols, screen=universe)


def minute_pipeline() -> Pipeline:
    """Intraday pipeline shown live by the demo (about 40 graph nodes)."""
    r = close.pct_change(1)
    liquid: Filter = volume.rolling_mean(30) > 0
    vwap30 = dollar.rolling_sum(30) / volume.rolling_sum(30)
    mom30 = close.pct_change(30)
    vol30 = r.rolling_std(30)
    score = (
        mom30.zscore(groupby=industry, mask=liquid) + signal.zscore(mask=liquid) - vol30.zscore(mask=liquid)
    )
    cols = {
        "close": close,
        "ret1": r,
        "mom30": mom30,
        "vol30": vol30,
        "vwap_dev": close / vwap30 - 1.0,
        "rel_vol": volume / volume.rolling_mean(60),
        "range30": (high.rolling_max(30) - low.rolling_min(30)) / close,
        "ewm_ret": r.ewm_mean(10.0),
        "score": score,
        "rank": score.rank(mask=liquid, ascending=False),
        "long": where(score.top(10, mask=liquid), 1.0, 0.0),
    }
    return Pipeline(cols, screen=liquid)


def crypto_pipeline() -> Pipeline:
    """24/7 minute pipeline for the crypto majors (beta to BTC via the cross-section)."""
    r = close.pct_change(1)
    btc = r - r.demean()  # with 3 names this is the equal-weight crypto market
    return Pipeline(
        {
            "close": close,
            "ret1": r,
            "mom60": close.pct_change(60),
            "vol60": r.rolling_std(60),
            "beta240": r.rolling_beta(btc, 240),
            "dollar60": dollar.rolling_sum(60),
            "mom_rank": close.pct_change(60).rank(ascending=False),
        }
    )


def wide_pipeline(n_factors: int = 100) -> Pipeline:
    """The 100-factor benchmark pipeline from ``benchmarks/bench_minute.py`` (with industries)."""
    c, v, h, lo = close, volume, high, low
    r = c.pct_change(1)
    liquid = v.rolling_mean(30) > 0
    windows = [5, 10, 15, 20, 30, 60, 90, 120]
    gens = [
        lambda w: c.pct_change(w),
        lambda w: r.rolling_std(w),
        lambda w: c / c.rolling_mean(w) - 1.0,
        lambda w: v.rolling_mean(w),
        lambda w: (c - lo.rolling_min(w)) / (h.rolling_max(w) - lo.rolling_min(w)),
        lambda w: c.ts_rank(w),
        lambda w: r.rolling_corr(v.pct_change(1), w),
        lambda w: r.ewm_mean(w / 2),
        lambda w: c.pct_change(w).zscore(mask=liquid, groupby=industry),
        lambda w: r.rolling_std(w).rank(mask=liquid),
        lambda w: (r.rolling_mean(w) / r.rolling_std(w)).demean(groupby=industry),
        lambda w: r.rolling_beta(r.demean(), w),
    ]
    cols: dict = {}
    i = 0
    while len(cols) < n_factors:
        g = gens[i % len(gens)]
        w = windows[(i // len(gens)) % len(windows)]
        cols[f"f{len(cols):03d}"] = g(w)
        i += 1
    return Pipeline(cols, screen=liquid)


def event_pipeline() -> Pipeline:
    """Per-asset time-series pipeline for event (tick/volume/dollar) bars."""
    r = close.pct_change(1)
    return Pipeline(
        {
            "ret": r,
            "vol20": r.rolling_std(20),
            "mom10": close.pct_change(10),
            "ewm": r.ewm_mean(5.0),
            "logc": close.log(),
        }
    )
