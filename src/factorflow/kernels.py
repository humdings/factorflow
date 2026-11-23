"""Numeric kernels.

Every time-series kernel is written **once**, as a function of a block of input rows
``x`` of shape ``(T_in, n)`` and a window length ``W``. It returns the ``T_in - W + 1``
valid output rows. The batch executor calls it on the whole history, and the live
executor calls it on exactly ``W`` rows (the sealed tail plus the open row). Both run
the *same* sequence of elementwise IEEE operations (fixed left-to-right accumulation
over the window, no pairwise or BLAS reductions), so a row computed live is bit-for-bit
identical to the same row in batch.

Cross-sectional kernels take one row ``(n,)`` and are applied row by row in batch, so the
same argument holds.

NaN policy: a time-series window containing any NaN yields NaN (Pipeline semantics).
Cross-sectional kernels drop NaN, a false mask, and group codes below 0, then
broadcast the group statistic back onto the assets that entered it. Recursive
kernels (exponentially weighted, cumulative) keep the previous value when the
new observation is NaN, and start at the first finite observation.
"""

from __future__ import annotations

import math

import numpy as np

F64 = np.float64


def _views(x: np.ndarray, w: int) -> list[np.ndarray]:
    """Lagged windows of ``x``, oldest first.

    ``views[k]`` is ``x[k : k + T_out]``. Adding those views left to right is the
    same sequence of additions a live call makes on one window of length ``w``,
    which is why a sealed row matches batch.
    """
    t_out = x.shape[0] - w + 1
    return [x[k : k + t_out] for k in range(w)]


def wsum(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling sum over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values. ``T`` is the number of rows (bars) and ``N`` the number of assets.
    w : int
        Window length in bars. Must be at least 1, and at most ``T``.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Sum of each window. A window that contains a NaN is NaN, because NaN poisons
        the sum. The sum is accumulated oldest to newest, one addition at a time.
    """
    v = _views(x, w)
    acc = v[0].astype(F64, copy=True)
    for k in range(1, w):
        acc += v[k]
    return acc


def wmean(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling mean over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        ``wsum(x, w) / w``. A window that contains a NaN is NaN.
    """
    return wsum(x, w) / w


def wvar(x: np.ndarray, w: int, ddof: int = 1) -> np.ndarray:
    """Return the rolling variance over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars. Needs ``w > ddof``.
    ddof : int, optional
        Delta degrees of freedom. The divisor is ``w - ddof``. Default is 1
        (the sample variance).

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Variance of each window. NaN if the window contains a NaN or the divisor
        is not positive.

    Notes
    -----
    The mean is the rolling sum divided by ``w``, and the squared deviations are
    summed oldest to newest. That order matches a live call on the same window.
    """
    v = _views(x, w)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        m = wsum(x, w) / w
        d = v[0] - m
        ss = d * d
        for k in range(1, w):
            d = v[k] - m
            ss += d * d
        return ss / (w - ddof) if w - ddof > 0 else np.full_like(ss, np.nan)


def wstd(x: np.ndarray, w: int, ddof: int = 1) -> np.ndarray:
    """Return the rolling standard deviation over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.
    ddof : int, optional
        Passed to :func:`wvar`. Default is 1.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Square root of the rolling variance.
    """
    return np.sqrt(wvar(x, w, ddof))


def wmin(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling minimum over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Minimum of each window. NaN if the window contains a NaN (``np.minimum``
        propagates it).
    """
    v = _views(x, w)
    acc = v[0].astype(F64, copy=True)
    for k in range(1, w):
        np.minimum(acc, v[k], out=acc)
    return acc


def wmax(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling maximum over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Maximum of each window. NaN if the window contains a NaN.
    """
    v = _views(x, w)
    acc = v[0].astype(F64, copy=True)
    for k in range(1, w):
        np.maximum(acc, v[k], out=acc)
    return acc


def wtsrank(x: np.ndarray, w: int) -> np.ndarray:
    """Time-series rank of the newest value inside each window.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Fraction of the window that is less than or equal to the newest value,
        in ``(1/w, ..., 1)``. NaN if the window contains a NaN. Ties with the
        newest value count as less than or equal, so a constant window ranks 1.
    """
    v = _views(x, w)
    last = v[-1]
    cnt = np.zeros(last.shape, dtype=F64)
    nan = np.isnan(last)
    for k in range(w):
        cnt += v[k] <= last
        nan |= np.isnan(v[k])
    out = cnt / w
    out[nan] = np.nan
    return out


def wcov(a: np.ndarray, b: np.ndarray, w: int, ddof: int = 1) -> np.ndarray:
    """Return the rolling covariance of two series.

    Parameters
    ----------
    a, b : ndarray of shape (T, N)
        Paired inputs. Compared elementwise; column ``j`` of ``a`` is paired
        with column ``j`` of ``b``.
    w : int
        Window length in bars.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Covariance of each window. NaN if either window contains a NaN.
    """
    va, vb = _views(a, w), _views(b, w)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ma, mb = wsum(a, w) / w, wsum(b, w) / w
        acc = (va[0] - ma) * (vb[0] - mb)
        for k in range(1, w):
            acc += (va[k] - ma) * (vb[k] - mb)
        return acc / (w - ddof)


def wcorr(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling Pearson correlation of two series.

    Parameters
    ----------
    a, b : ndarray of shape (T, N)
        Paired inputs, column by column.
    w : int
        Window length in bars. A constant window has a zero standard deviation,
        and the correlation is NaN there.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        ``wcov(a, b, w) / (wstd(a, w) * wstd(b, w))`` with ``ddof`` 1.
    """
    c = wcov(a, b, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return c / (wstd(a, w) * wstd(b, w))


def wshift(x: np.ndarray, w: int) -> np.ndarray:
    """Value from ``w - 1`` rows earlier.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        One more than the lag. A lag of ``n`` bars is ``w = n + 1``, so the
        output lines up with the other windowed kernels (``T - w + 1`` rows).

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        ``x`` truncated to the rows that have a full lag. This is a copy.
    """
    return x[: x.shape[0] - w + 1].copy()


def ewm_alpha(halflife: float) -> float:
    """Smoothing weight for an exponentially weighted mean.

    Parameters
    ----------
    halflife : float
        Half-life in bars. Must be positive. A half-life of 1 means the previous
        value is halved each bar.

    Returns
    -------
    float
        ``1 - exp(-ln(2) / halflife)``, in ``(0, 1)``.
    """
    return 1.0 - math.exp(-math.log(2.0) / halflife)


def ewm_step(x: np.ndarray, prev: np.ndarray, alpha: float) -> np.ndarray:
    """Advance an exponentially weighted mean by one row.

    Parameters
    ----------
    x : ndarray of shape (N,)
        The new observation. A NaN keeps ``prev`` (the mean does not jump to missing).
    prev : ndarray of shape (N,)
        The previous mean. A NaN starts the mean at ``x``.
    alpha : float
        Weight on ``x``, from :func:`ewm_alpha`. The previous mean keeps ``1 - alpha``.

    Returns
    -------
    ndarray of shape (N,)
        The updated mean.
    """
    upd = alpha * x + (1.0 - alpha) * prev
    out = np.where(np.isnan(prev), x, upd)
    return np.where(np.isnan(x), prev, out)


def ewm_std_step(x: np.ndarray, prev: np.ndarray | None, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance an exponentially weighted standard deviation by one row.

    Parameters
    ----------
    x : ndarray of shape (N,)
        The new observation. A NaN keeps the previous mean and variance.
    prev : ndarray of shape (3, N) or None
        Row 0 is the previous mean, row 1 the previous variance, row 2 the
        count of finite observations seen so far. ``None`` means the series
        has not started. A count of 1 stores variance 0, which is the seed
        for the next update, not a reported standard deviation.
    alpha : float
        Weight on ``x``, from :func:`ewm_alpha`. The previous moments keep
        ``1 - alpha``.

    Returns
    -------
    std : ndarray of shape (N,)
        Square root of the variance. NaN until two finite observations have
        been seen for that asset.
    state : ndarray of shape (3, N)
        Updated mean, variance, and count.

    Notes
    -----
    The mean uses the same recurrence as :func:`ewm_step`. The variance is the
    bias-included (population) update, with ``delta = x - mean`` and ``mean``
    the previous mean:

    ``mean' = mean + alpha * delta``

    ``var' = (1 - alpha) * var + alpha * delta * (x - mean')``

    ``var`` starts at 0 after the first finite observation. This matches
    ``Series.ewm(halflife, adjust=False, ignore_na=True).std(bias=True)``
    after that first observation. The first observation itself is NaN here,
    because a one-point dispersion is not a standard deviation.
    """
    x = np.asarray(x, dtype=F64)
    n = x.shape[0]
    if prev is None:
        mean = np.full(n, np.nan, dtype=F64)
        var = np.zeros(n, dtype=F64)
        nobs = np.zeros(n, dtype=F64)
    else:
        mean = np.array(prev[0], dtype=F64, copy=True)
        var = np.array(prev[1], dtype=F64, copy=True)
        nobs = np.array(prev[2], dtype=F64, copy=True)
    finite = np.isfinite(x)
    start = finite & (nobs == 0)
    mean = np.where(start, x, mean)
    var = np.where(start, 0.0, var)
    nobs = np.where(start, 1.0, nobs)
    upd = finite & (nobs >= 1) & ~start
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        delta = x - mean
        mean_new = mean + alpha * delta
        var_new = (1.0 - alpha) * var + alpha * delta * (x - mean_new)
    mean = np.where(upd, mean_new, mean)
    var = np.where(upd, np.maximum(var_new, 0.0), var)
    nobs = np.where(upd, nobs + 1.0, nobs)
    std = np.where(nobs >= 2.0, np.sqrt(var), np.nan).astype(F64, copy=False)
    return std, np.vstack((mean, var, nobs))


def cum_step(x: np.ndarray, prev: np.ndarray | None, how: str) -> np.ndarray:
    """Advance a cumulative reduction by one row.

    Parameters
    ----------
    x : ndarray of shape (N,)
        The new observation. A NaN keeps ``prev``. The running value starts
        at the first finite observation.
    prev : ndarray of shape (N,) or None
        The previous cumulative value. ``None`` and NaN both mean "not
        started". Inf is a real running value (an overflow) and is kept.
    how : str
        ``sum``, ``prod``, ``min``, or ``max``.

    Returns
    -------
    ndarray of shape (N,)
        The updated cumulative value.

    Raises
    ------
    ValueError
        If ``how`` is not one of the four reductions.
    """
    x = np.asarray(x, dtype=F64)
    prev = np.full(x.shape, np.nan, dtype=F64) if prev is None else np.asarray(prev, dtype=F64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if how == "sum":
            mixed = prev + x
        elif how == "prod":
            mixed = prev * x
        elif how == "min":
            mixed = np.minimum(prev, x)
        elif how == "max":
            mixed = np.maximum(prev, x)
        else:
            raise ValueError(f"unknown cumulative reduction {how!r}")
    start = np.isnan(prev) & np.isfinite(x)
    cont = ~np.isnan(prev) & np.isfinite(x)
    return np.where(start, x, np.where(cont, mixed, prev)).astype(F64, copy=False)


def _power_sums(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left-to-right sums of centered powers ``d**2``, ``d**3``, and ``d**4``.

    The mean is the rolling sum divided by ``w``. Deviations are accumulated
    oldest to newest, the same order a live call uses on one window.
    """
    v = _views(x, w)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        m = wsum(x, w) / w
        d = v[0] - m
        d2 = d * d
        s2 = d2.copy()
        s3 = d2 * d
        s4 = d2 * d2
        nan = np.isnan(v[0])
        for k in range(1, w):
            d = v[k] - m
            d2 = d * d
            s2 += d2
            s3 += d2 * d
            s4 += d2 * d2
            nan |= np.isnan(v[k])
        return s2, s3, s4, nan


def _skew_from_sums(sum2: np.ndarray, sum3: np.ndarray, n: float) -> np.ndarray:
    """Bias-adjusted Fisher-Pearson skewness from sums of centered powers.

    ``sum2`` and ``sum3`` are sums, not means. The adjustment is
    ``sqrt(n*(n-1)) / (n-2) * m3 / m2**1.5`` with ``m2 = sum2/n``. A
    non-positive second moment is NaN. ``n`` must be at least 3.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        m2 = sum2 / n
        m3 = sum3 / n
        g1 = m3 / (m2**1.5)
        out = math.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    out = np.asarray(out, dtype=F64)
    out[~(m2 > 0)] = np.nan
    return out


def _kurtosis_from_sums(sum2: np.ndarray, sum4: np.ndarray, n: float) -> np.ndarray:
    """Bias-adjusted excess kurtosis from sums of centered powers.

    ``g2 = m4/m2**2 - 3`` with moments divided by ``n``, then
    ``((n-1)/((n-2)*(n-3))) * ((n+1)*g2 + 6)``. A normal sample is 0.
    A non-positive second moment is NaN. ``n`` must be at least 4.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        m2 = sum2 / n
        m4 = sum4 / n
        g2 = m4 / (m2**2) - 3.0
        out = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
    out = np.asarray(out, dtype=F64)
    out[~(m2 > 0)] = np.nan
    return out


def _stacked(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Window values as ``(T_out, N, w)``, plus a mask of windows that contain a NaN."""
    stacked = np.stack(_views(x, w), axis=-1)
    return stacked, np.isnan(stacked).any(axis=-1)


def wmedian(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling median over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Median of each window. An even length averages the two central order
        statistics. A window that contains a NaN is NaN.
    """
    stacked, bad = _stacked(x, w)
    with np.errstate(invalid="ignore"):
        out = np.median(stacked, axis=-1).astype(F64, copy=False)
    out[bad] = np.nan
    return out


def wquantile(x: np.ndarray, w: int, q: float) -> np.ndarray:
    """Return the rolling quantile over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.
    q : float
        Fraction in ``[0, 1]``. ``0`` is the minimum and ``1`` is the maximum.
        ``0.5`` is the median. This is a quantile, not a 0-100 percentile.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Linear quantile: the value at index ``q * (w - 1)`` in the sorted
        window, interpolated between the surrounding order statistics. A
        window that contains a NaN is NaN.
    """
    stacked, bad = _stacked(x, w)
    with np.errstate(invalid="ignore"):
        out = np.quantile(stacked, q, axis=-1, method="linear").astype(F64, copy=False)
    out[bad] = np.nan
    return out


def wskew(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling bias-adjusted skewness over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars. A result needs ``w >= 3``.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Fisher-Pearson skewness with the sample adjustment
        ``sqrt(w*(w-1))/(w-2)``. A symmetric window is 0. NaN if the window
        contains a NaN, has no dispersion, or is shorter than 3.
    """
    if w < 3:
        return np.full((x.shape[0] - w + 1, x.shape[1]), np.nan, dtype=F64)
    s2, s3, _, nan = _power_sums(x, w)
    out = _skew_from_sums(s2, s3, float(w))
    out[nan] = np.nan
    return out


def wkurtosis(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling bias-adjusted excess kurtosis over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars. A result needs ``w >= 4``.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Excess kurtosis (a normal window is 0) with the sample adjustment
        ``((w-1)/((w-2)*(w-3))) * ((w+1)*g2 + 6)``. NaN if the window
        contains a NaN, has no dispersion, or is shorter than 4.
    """
    if w < 4:
        return np.full((x.shape[0] - w + 1, x.shape[1]), np.nan, dtype=F64)
    s2, _, s4, nan = _power_sums(x, w)
    out = _kurtosis_from_sums(s2, s4, float(w))
    out[nan] = np.nan
    return out


def wmad(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling median absolute deviation over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Median of ``|x - median(x)|`` inside the window. A window that
        contains a NaN is NaN. A one-point window is 0.
    """
    stacked, bad = _stacked(x, w)
    with np.errstate(invalid="ignore"):
        med = np.median(stacked, axis=-1)
        out = np.median(np.abs(stacked - med[..., None]), axis=-1).astype(F64, copy=False)
    out[bad] = np.nan
    return out


def wcount(x: np.ndarray, w: int) -> np.ndarray:
    """Return ``w`` when the window is complete, else NaN.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        The float ``w`` if every value in the window is finite, otherwise
        NaN. This is the same missing-window rule as the other rolling
        reductions, not a count of the non-missing entries inside the window.
    """
    _, bad = _stacked(x, w)
    out = np.full(bad.shape, float(w), dtype=F64)
    out[bad] = np.nan
    return out


def wprod(x: np.ndarray, w: int) -> np.ndarray:
    """Return the rolling product over ``w`` rows.

    Parameters
    ----------
    x : ndarray of shape (T, N)
        Input values.
    w : int
        Window length in bars.

    Returns
    -------
    ndarray of shape (T - w + 1, N)
        Product of each window, multiplied oldest to newest. A window that
        contains a NaN is NaN. Overflow is inf.
    """
    v = _views(x, w)
    acc = v[0].astype(F64, copy=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for k in range(1, w):
            acc *= v[k]
    return acc


# ------------------------------------------------------------------ cross-sectional
def _valid_groups(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None):
    valid = ~np.isnan(v)
    if mask is not None:
        valid &= mask
    if groups is None:
        g = np.zeros(v.shape[0], dtype=np.int64)
    else:
        g = groups.astype(np.int64, copy=False)
        valid &= g >= 0
    return valid, g


def cs_rank(
    v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None, ascending: bool = True
) -> np.ndarray:
    """Ordinal rank within each group.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section of values.
    mask : ndarray of bool, shape (N,), or None
        Positions that take part. ``None`` keeps every finite value.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing and are
        not ranked.
    ascending : bool, optional
        ``True`` gives rank 1 to the smallest value. Default is ``True``.

    Returns
    -------
    ndarray of shape (N,)
        Ranks ``1 .. k`` inside each group, as float. Ties keep asset order
        (the earlier column wins the lower rank). Invalid entries are NaN.
    """
    out = np.full(v.shape[0], np.nan)
    valid, g = _valid_groups(v, mask, groups)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    vals = v[idx] if ascending else -v[idx]
    gi = g[idx]
    order = np.lexsort((idx, vals, gi))  # group, then value, then column index (stable ties)
    sg = gi[order]
    pos = np.arange(order.size)
    starts = np.r_[0, np.flatnonzero(sg[1:] != sg[:-1]) + 1]
    first = np.repeat(starts, np.diff(np.r_[starts, order.size]))
    out[idx[order]] = (pos - first + 1).astype(F64)
    return out


def _group_moments(v: np.ndarray, valid: np.ndarray, g: np.ndarray):
    gv = g[valid]
    x = v[valid]
    ng = int(gv.max()) + 1 if gv.size else 0
    cnt = np.bincount(gv, minlength=ng).astype(F64)
    s = np.bincount(gv, weights=x, minlength=ng)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / cnt
    return gv, x, cnt, mean


def cs_demean(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Subtract the group mean.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the mean and receive a result.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        ``v`` minus its group's mean. Invalid entries are NaN. A group of one
        is demeaned to 0.
    """
    out = np.full(v.shape[0], np.nan)
    valid, g = _valid_groups(v, mask, groups)
    if not valid.any():
        return out
    gv, x, _cnt, mean = _group_moments(v, valid, g)
    out[valid] = x - mean[gv]
    return out


def cs_zscore(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Standardize by the group mean and sample standard deviation.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the moments and receive a result.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        ``(v - mean) / std`` with the sample standard deviation (divisor
        ``count - 1``). A group with fewer than two valid values, or with zero
        dispersion, is NaN.
    """
    out = np.full(v.shape[0], np.nan)
    valid, g = _valid_groups(v, mask, groups)
    if not valid.any():
        return out
    gv, x, cnt, mean = _group_moments(v, valid, g)
    dev = x - mean[gv]
    ss = np.bincount(gv, weights=dev * dev, minlength=cnt.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(ss / (cnt - 1.0))
        z = dev / sd[gv]
    z[cnt[gv] < 2] = np.nan
    out[valid] = z
    return out


def _broadcast(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None, fn) -> np.ndarray:
    """Apply ``fn`` to each group's valid values and write that scalar back.

    ``fn`` receives a 1-d array in column order. Assets that are NaN, masked
    out, or in a missing group stay NaN and do not enter ``fn``.
    """
    out = np.full(v.shape[0], np.nan, dtype=F64)
    valid, g = _valid_groups(v, mask, groups)
    if not valid.any():
        return out
    idx = np.flatnonzero(valid)
    codes = g[idx]
    order = np.argsort(codes, kind="mergesort")
    codes = codes[order]
    idx = idx[order]
    vals = v[idx]
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    ends = np.r_[starts[1:], codes.size]
    for a, b in zip(starts, ends, strict=True):
        out[idx[a:b]] = fn(vals[a:b])
    return out


def _sample_var(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return np.nan
    mean = float(x.sum() / n)
    return float(((x - mean) ** 2).sum() / (n - 1))


def _cs_skew(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return np.nan
    mean = float(x.sum() / n)
    d = x - mean
    sums = _skew_from_sums(np.array([(d * d).sum()]), np.array([(d * d * d).sum()]), float(n))
    return float(sums[0])


def _cs_kurtosis(x: np.ndarray) -> float:
    n = x.size
    if n < 4:
        return np.nan
    mean = float(x.sum() / n)
    d2 = (x - mean) ** 2
    sums = _kurtosis_from_sums(np.array([d2.sum()]), np.array([(d2 * d2).sum()]), float(n))
    return float(sums[0])


def cs_mean(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group mean.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the mean and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The mean of the valid values in the group, repeated on each of them.
        Invalid entries are NaN.
    """
    return _broadcast(v, mask, groups, lambda x: float(x.sum() / x.size))


def cs_sum(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group sum.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the sum and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The sum of the valid values, in column order, repeated on each of them.
    """
    return _broadcast(v, mask, groups, lambda x: float(x.sum()))


def cs_median(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group median.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the median and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The median of the valid values. An even count averages the two central
        order statistics. Invalid entries are NaN.
    """
    return _broadcast(v, mask, groups, lambda x: float(np.median(x)))


def cs_var(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group sample variance.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the variance and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        Sample variance, divisor ``count - 1``. A group with fewer than two
        valid values is NaN.
    """
    return _broadcast(v, mask, groups, _sample_var)


def cs_std(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group sample standard deviation.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the standard deviation and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        Square root of the sample variance. A group with fewer than two valid
        values is NaN. Zero dispersion is 0.
    """
    return _broadcast(v, mask, groups, lambda x: float(np.sqrt(_sample_var(x))))


def cs_min(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group minimum.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the minimum and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The smallest valid value in the group, repeated on each of them.
    """
    return _broadcast(v, mask, groups, lambda x: float(x.min()))


def cs_max(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group maximum.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the maximum and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The largest valid value in the group, repeated on each of them.
    """
    return _broadcast(v, mask, groups, lambda x: float(x.max()))


def cs_quantile(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None, q: float) -> np.ndarray:
    """Broadcast the group quantile.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the quantile and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.
    q : float
        Fraction in ``[0, 1]``. Linear interpolation, the same rule as
        :func:`wquantile`. ``0.5`` is the median.

    Returns
    -------
    ndarray of shape (N,)
        The quantile of the valid values, repeated on each of them.
    """
    return _broadcast(v, mask, groups, lambda x: float(np.quantile(x, q, method="linear")))


def cs_skew(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group bias-adjusted skewness.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the skewness and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The same Fisher-Pearson adjustment as :func:`wskew`. A group with
        fewer than three valid values, or with no dispersion, is NaN.
    """
    return _broadcast(v, mask, groups, _cs_skew)


def cs_kurtosis(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group bias-adjusted excess kurtosis.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the kurtosis and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The same excess-kurtosis adjustment as :func:`wkurtosis`. A normal
        group is 0. A group with fewer than four valid values, or with no
        dispersion, is NaN.
    """
    return _broadcast(v, mask, groups, _cs_kurtosis)


def cs_mad(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group median absolute deviation.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the deviation and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        Median of ``|x - median(x)|`` over the valid values. A group of one
        is 0.
    """

    def mad(x: np.ndarray) -> float:
        med = float(np.median(x))
        return float(np.median(np.abs(x - med)))

    return _broadcast(v, mask, groups, mad)


def cs_count(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast how many values entered the group.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that are counted and receive the count.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The number of valid values in the group, as a float, repeated on each
        of them. Invalid entries are NaN, so a masked asset is not counted
        and does not receive the count.
    """
    return _broadcast(v, mask, groups, lambda x: float(x.size))


def cs_product(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
    """Broadcast the group product.

    Parameters
    ----------
    v : ndarray of shape (N,)
        One cross-section.
    mask : ndarray of bool, shape (N,), or None
        Positions that enter the product and receive it.
    groups : ndarray of int, shape (N,), or None
        Group codes. ``None`` is one group. Codes below 0 are missing.

    Returns
    -------
    ndarray of shape (N,)
        The product of the valid values, multiplied in column order. Overflow
        is inf.
    """

    def product(x: np.ndarray) -> float:
        acc = 1.0
        for value in x:
            acc *= float(value)
        return acc

    return _broadcast(v, mask, groups, product)
