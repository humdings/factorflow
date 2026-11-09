import os

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

import factorflow as ff

settings.register_profile(
    "default",
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile(
    "ci",
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


def full_pipeline() -> ff.Pipeline:
    """Touches every kernel: elementwise, time-series, recursive, cross-sectional, masks, groups."""
    c, v, h, lo = ff.BarData.close, ff.BarData.volume, ff.BarData.high, ff.BarData.low
    sector = ff.Column("sector", ff.Classifier)
    r = c.pct_change(1)
    liquid = v.rolling_mean(3) > 0
    cols = {
        "ret5": c.pct_change(5),
        "mom": r.rolling_sum(4),
        "mean": c.rolling_mean(6),
        "vol": r.rolling_std(5),
        "var": r.rolling_var(3),
        "lo": lo.rolling_min(4),
        "hi": h.rolling_max(4),
        "tsr": c.ts_rank(5),
        "cov": r.rolling_cov(v, 4),
        "corr": c.rolling_corr(v, 5),
        "ewm": r.ewm_mean(3.0),
        "logc": c.log(),
        "delta": v.delta(2),
        "z": r.zscore(mask=liquid),
        "zg": r.zscore(groupby=sector),
        "dm": c.demean(groupby=sector, mask=liquid),
        "rk": r.rank(mask=liquid),
        "rkg": c.rank(groupby=sector, ascending=False),
        "ew2": ff.where(c > c.shift(1), v, -v).ewm_mean(2.0),
        "top": ff.where((r.top(2, groupby=sector) & ff.BarData.has_trade).shift(1) | ~liquid, 1.0, 0.0),
        "fill": r.fillna(0.0) * 2.0 + 1.0,
    }
    return ff.Pipeline(cols, screen=c.notnull())


@pytest.fixture
def pipeline() -> ff.Pipeline:
    return full_pipeline()


def assert_rows_equal(live_vals: dict, batch_vals: dict, row: int, ctx: str = "") -> None:
    for k, v in live_vals.items():
        b = batch_vals[k][row]
        assert np.array_equal(v, b, equal_nan=True), f"{ctx} column {k!r} differs at row {row}:\n{v}\n{b}"
