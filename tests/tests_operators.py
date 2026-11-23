"""Hand-worked checks for the elementwise, rolling, cumulative, and cross-sectional operators."""

import math

import numpy as np
import pandas as pd
import pytest

import factorflow as ff
from factorflow import kernels as K

SQRT2 = math.sqrt(2.0)
SQRT_675 = math.sqrt(6.75)  # sqrt(27/4), the third ewm step below


def _panel(columns: dict[str, np.ndarray]) -> ff.Panel:
    sample = next(iter(columns.values()))
    t, n = sample.shape
    return ff.Panel(np.arange(t), tuple(f"a{i}" for i in range(n)), columns)


def _run(columns: dict[str, np.ndarray], **outputs) -> dict[str, np.ndarray]:
    return ff.BatchEngine().run(ff.Pipeline(outputs), _panel(columns)).values


def _col(name: str = "x") -> ff.Factor:
    return ff.Column(name)


def test_exp_matches_hand_values_and_numpy():
    # exp(0) = 1, exp(1) = e, exp(2) = e^2, exp(-1) = 1/e. NaN stays NaN.
    x = np.array([[0.0, 1.0], [2.0, -1.0], [np.nan, 0.0]])
    out = _run({"x": x}, e=_col().exp())["e"]
    hand = np.array(
        [
            [1.0, 2.718281828459045],
            [7.38905609893065, 0.3678794411714423],
            [np.nan, 1.0],
        ]
    )
    np.testing.assert_allclose(out, hand, rtol=1e-15)
    np.testing.assert_allclose(out, np.exp(x), rtol=0, atol=0, equal_nan=True)
    # log then exp returns the positive input.
    back = _run({"x": np.array([[math.e], [4.0]])}, y=_col().log().exp())["y"]
    np.testing.assert_allclose(back[:, 0], [math.e, 4.0])


def test_rolling_median_odd_even_and_nan():
    # Odd window [1, 3, 2] -> 2. Even window averages the two central order stats.
    x = np.array([[1.0], [3.0], [2.0], [10.0], [6.0]])
    odd = _run({"x": x}, m=_col().rolling_median(3))["m"][:, 0]
    # windows ending at t=2,3,4: (1,3,2)=2, (3,2,10)=3, (2,10,6)=6
    np.testing.assert_array_equal(odd, [np.nan, np.nan, 2.0, 3.0, 6.0])
    even = _run({"x": x}, m=_col().rolling_median(2))["m"][:, 0]
    # (1+3)/2=2, (3+2)/2=2.5, (2+10)/2=6, (10+6)/2=8
    np.testing.assert_array_equal(even, [np.nan, 2.0, 2.5, 6.0, 8.0])
    poisoned = x.copy()
    poisoned[2, 0] = np.nan
    bad = _run({"x": poisoned}, m=_col().rolling_median(3))["m"][:, 0]
    # the NaN sits in the three windows that end at t=2,3,4
    assert np.isnan(bad[2:]).all()
    np.testing.assert_array_equal(bad[:2], [np.nan, np.nan])


def test_rolling_quantile_is_linear_interpolation():
    # Sorted window 1,2,3,4. Index q*(4-1). q=0.25 -> 0.75 of the way from 1 to 2 = 1.75.
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = _run(
        {"x": x},
        q0=_col().rolling_quantile(4, 0),
        q25=_col().rolling_quantile(4, 0.25),
        q50=_col().rolling_quantile(4, 0.5),
        q100=_col().rolling_quantile(4, 1),
    )
    assert out["q0"][-1, 0] == 1.0
    assert out["q25"][-1, 0] == 1.75
    assert out["q50"][-1, 0] == 2.5  # average of 2 and 3, same as the median
    assert out["q100"][-1, 0] == 4.0
    np.testing.assert_allclose(out["q25"][-1], K.wquantile(x, 4, 0.25)[-1])
    with pytest.raises(ValueError):
        _col().rolling_quantile(4, 1.5)
    with pytest.raises(ValueError):
        _col().rolling_quantile(4, True)  # type: ignore[arg-type]


def test_rolling_moments_match_hand_values():
    # Product 2*3*4 = 24. Count is the window length, or NaN if any value is missing.
    x = np.array([[2.0], [3.0], [4.0], [np.nan]])
    out = _run({"x": x}, p=_col().rolling_product(3), c=_col().rolling_count(3))
    np.testing.assert_array_equal(out["p"][:, 0], [np.nan, np.nan, 24.0, np.nan])
    np.testing.assert_array_equal(out["c"][:, 0], [np.nan, np.nan, 3.0, np.nan])

    # Skew of [1, 1, 1, 4]. mean = 1.75. deviations -0.75, -0.75, -0.75, 2.25.
    # sum d^2 = 6.75, sum d^3 = 10.125. The adjusted skewness simplifies to 2.
    skew_x = np.array([[1.0], [1.0], [1.0], [4.0]])
    skew = _run({"x": skew_x}, s=_col().rolling_skew(4))["s"][-1, 0]
    assert skew == pytest.approx(2.0)
    assert skew == pytest.approx(float(pd.Series([1.0, 1.0, 1.0, 4.0]).skew()))

    # Excess kurtosis of [1, 2, 3, 4] is -1.2.
    # mean 2.5; sum d^2 = 5; sum d^4 = 10.25; g2 = -1.36; adjusted = -1.2.
    kurt_x = np.array([[1.0], [2.0], [3.0], [4.0]])
    kurt = _run({"x": kurt_x}, k=_col().rolling_kurtosis(4))["k"][-1, 0]
    assert kurt == pytest.approx(-1.2)
    assert kurt == pytest.approx(float(pd.Series([1.0, 2.0, 3.0, 4.0]).kurt()))

    # MAD of [1, 2, 3, 4]: median 2.5, absolute deviations 1.5, 0.5, 0.5, 1.5, median 1.
    mad = _run({"x": kurt_x}, m=_col().rolling_mad(4))["m"][-1, 0]
    assert mad == 1.0

    # A constant window has no dispersion, so skew and kurtosis are missing.
    flat = np.ones((4, 1))
    gone = _run({"x": flat}, s=_col().rolling_skew(4), k=_col().rolling_kurtosis(4))
    assert np.isnan(gone["s"][-1, 0]) and np.isnan(gone["k"][-1, 0])


def test_rolling_reference_matches_per_window_numpy():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 3))
    x[4, 1] = np.nan
    got = _run(
        {"x": x},
        med=_col().rolling_median(4),
        q=_col().rolling_quantile(4, 0.25),
        mad=_col().rolling_mad(4),
        prod=_col().rolling_product(3),
    )

    def per_window(w, fn):
        out = np.full(x.shape, np.nan)
        for t in range(w - 1, x.shape[0]):
            block = x[t - w + 1 : t + 1]
            bad = np.isnan(block).any(axis=0)
            value = fn(block)
            value[bad] = np.nan
            out[t] = value
        return out

    np.testing.assert_allclose(got["med"], per_window(4, lambda a: np.median(a, axis=0)), equal_nan=True)
    np.testing.assert_allclose(
        got["q"], per_window(4, lambda a: np.quantile(a, 0.25, axis=0, method="linear")), equal_nan=True
    )
    np.testing.assert_allclose(
        got["mad"],
        per_window(4, lambda a: np.median(np.abs(a - np.median(a, axis=0)), axis=0)),
        equal_nan=True,
    )
    np.testing.assert_allclose(got["prod"], per_window(3, lambda a: np.prod(a, axis=0)), equal_nan=True)


def test_cumulative_skips_missing_and_starts_at_first_finite():
    x = np.array([[1.0], [np.nan], [3.0], [4.0]])
    # sum: 1, keep 1, 1+3=4, 4+4=8. min stays 1.
    summed = _run({"x": x}, s=_col().cumsum(), lo=_col().cummin())
    np.testing.assert_array_equal(summed["s"][:, 0], [1.0, 1.0, 4.0, 8.0])
    np.testing.assert_array_equal(summed["lo"][:, 0], [1.0, 1.0, 1.0, 1.0])
    # A leading gap does not start the run. product: 2, 2*3=6, keep 6, 6*4=24.
    # max: nan, 2, 3, 3, 4.
    y = np.array([[np.nan], [2.0], [3.0], [np.nan], [4.0]])
    ran = _run({"x": y}, p=_col().cumprod(), hi=_col().cummax())
    np.testing.assert_array_equal(ran["p"][:, 0], [np.nan, 2.0, 6.0, 6.0, 24.0])
    np.testing.assert_array_equal(ran["hi"][:, 0], [np.nan, 2.0, 3.0, 3.0, 4.0])
    finite = np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(_run({"x": finite}, s=_col().cumsum())["s"], np.cumsum(finite, axis=0))


def test_ewm_std_hand_recurrence_and_pandas():
    # halflife 1 -> alpha = 1/2.
    # t0: mean 1, variance not yet a deviation -> NaN
    # t1: delta=3-1=2, mean=2, var=(1/2)*0 + (1/2)*2*(3-2)=1, std=1
    # t2: delta=7-2=5, mean=4.5, var=(1/2)*1 + (1/2)*5*(7-4.5)=6.75
    x = np.array([[1.0], [3.0], [7.0]])
    out = _run({"x": x}, s=_col().ewm_std(1.0))["s"][:, 0]
    np.testing.assert_array_equal(out[:1], [np.nan])
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(SQRT_675)

    # A missing bar keeps the previous moments, so [1, nan, 3, 7] matches [1, 3, 7] on the finite steps.
    y = np.array([[1.0], [np.nan], [3.0], [7.0]])
    skipped = _run({"x": y}, s=_col().ewm_std(1.0))["s"][:, 0]
    np.testing.assert_array_equal(skipped[:2], [np.nan, np.nan])
    assert skipped[2] == pytest.approx(1.0)
    assert skipped[3] == pytest.approx(SQRT_675)

    ref = pd.Series(y[:, 0]).ewm(halflife=1, adjust=False, ignore_na=True).std(bias=True)
    # pandas reports 0 before the second finite point; the library reports NaN there.
    np.testing.assert_allclose(skipped[2:], ref.to_numpy()[2:])


def test_cross_section_broadcasts_hand_values():
    # One bar, four assets. Group 0 is 1 and 3. Group 1 is 10 and 2. -1 is missing.
    v = np.array([[1.0, 3.0, 10.0, 2.0, 9.0]])
    g = np.array([[0, 0, 1, 1, -1]])
    x = _col()
    group = ff.Column("g", ff.Classifier)
    out = _run(
        {"x": v, "g": g},
        mean=x.mean(groupby=group),
        sum=x.sum(groupby=group),
        med=x.median(groupby=group),
        var=x.var(groupby=group),
        std=x.std(groupby=group),
        lo=x.min(groupby=group),
        hi=x.max(groupby=group),
        n=x.count(groupby=group),
        prod=x.product(groupby=group),
    )
    # group 0 mean (1+3)/2 = 2, sum 4, median 2, sample var ((1+1)/1)=2, std sqrt(2)
    # group 1 mean 6, sum 12, median 6, var ((16+16)/1)=32, product 20
    np.testing.assert_allclose(out["mean"][0], [2, 2, 6, 6, np.nan])
    np.testing.assert_allclose(out["sum"][0], [4, 4, 12, 12, np.nan])
    np.testing.assert_allclose(out["med"][0], [2, 2, 6, 6, np.nan])
    np.testing.assert_allclose(out["var"][0], [2, 2, 32, 32, np.nan])
    np.testing.assert_allclose(out["std"][0], [SQRT2, SQRT2, math.sqrt(32), math.sqrt(32), np.nan])
    np.testing.assert_array_equal(out["lo"][0], [1, 1, 2, 2, np.nan])
    np.testing.assert_array_equal(out["hi"][0], [3, 3, 10, 10, np.nan])
    np.testing.assert_array_equal(out["n"][0], [2, 2, 2, 2, np.nan])
    np.testing.assert_array_equal(out["prod"][0], [3, 3, 20, 20, np.nan])


def test_cross_section_mask_quantile_skew_kurtosis_and_mad():
    # Eligible values 1, 3, 4 (the 2 is masked). mean = 8/3. median = 3.
    v = np.array([[1.0, 2.0, 3.0, 4.0]])
    mask = np.array([[True, False, True, True]])
    x = _col()
    screen = ff.Column("m", ff.Filter)
    out = _run(
        {"x": v, "m": mask},
        mean=x.mean(mask=screen),
        med=x.median(mask=screen),
        q=x.quantile(0, mask=screen),
    )
    assert out["mean"][0, 0] == pytest.approx(8 / 3)
    np.testing.assert_array_equal(out["med"][0], [3, np.nan, 3, 3])
    np.testing.assert_array_equal(out["q"][0], [1, np.nan, 1, 1])

    # Skew of [1, 1, 1, 4] is 2. Kurtosis of [1, 2, 3, 4] is -1.2. MAD is 1.
    skew = _run({"x": np.array([[1.0, 1.0, 1.0, 4.0]])}, s=_col().skew())["s"][0]
    np.testing.assert_allclose(skew, [2, 2, 2, 2])
    four = np.array([[1.0, 2.0, 3.0, 4.0]])
    shape = _run({"x": four}, k=_col().kurtosis(), mad=_col().mad())
    np.testing.assert_allclose(shape["k"][0], [-1.2, -1.2, -1.2, -1.2])
    np.testing.assert_array_equal(shape["mad"][0], [1, 1, 1, 1])

    # One asset: the mean is itself, the deviation statistics are missing, MAD is 0.
    alone = _run({"x": np.array([[5.0, np.nan]])}, mean=_col().mean(), std=_col().std(), mad=_col().mad())
    assert alone["mean"][0, 0] == 5.0
    assert np.isnan(alone["std"][0, 0])
    assert alone["mad"][0, 0] == 0.0


def test_live_matches_batch_including_open_revisions():
    x = np.array([[1.0, 4.0], [3.0, 1.0], [2.0, 8.0], [10.0, 2.0], [6.0, 5.0]])
    pipe = ff.Pipeline(
        {
            "exp": _col().exp(),
            "med": _col().rolling_median(3),
            "ewm": _col().ewm_std(1.0),
            "csum": _col().cumsum(),
            "xmed": _col().median(),
        }
    )
    panel = _panel({"x": x})
    batch = ff.BatchEngine().run(pipe, panel).values
    live = ff.LiveEngine(pipe, panel.head(2), grid=None)
    # Revise the open bar twice, then seal. The second revise must not commit the first.
    for i in range(2, x.shape[0]):
        live.set_open("x", x[i] + 1.0)
        live.revise()
        live.set_open("x", x[i])
        snap = live.seal(int(panel.times[i]))
        for name, values in snap.values.items():
            np.testing.assert_array_equal(values, batch[name][i], err_msg=name)


def test_window_and_halflife_validation():
    c = ff.Column("x")
    with pytest.raises(ValueError):
        c.rolling_median(0)
    with pytest.raises(ValueError):
        c.rolling_skew(2)
    with pytest.raises(ValueError):
        c.rolling_kurtosis(3)
    with pytest.raises(ValueError):
        c.ewm_std(0)
    with pytest.raises(ValueError):
        c.quantile(-0.1)
