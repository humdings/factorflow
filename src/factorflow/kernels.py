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
"""

from __future__ import annotations

import math

import numpy as np

F64 = np.float64


def _views(x: np.ndarray, w: int) -> list[np.ndarray]:
    """The W lagged views x[k : k + T_out] for k = 0..W-1 (oldest first)."""
    t_out = x.shape[0] - w + 1
    return [x[k : k + t_out] for k in range(w)]


def wsum(x: np.ndarray, w: int) -> np.ndarray:
    v = _views(x, w)
    acc = v[0].astype(F64, copy=True)
    for k in range(1, w):
        acc += v[k]
    return acc


def wmean(x: np.ndarray, w: int) -> np.ndarray:
    return wsum(x, w) / w


def wvar(x: np.ndarray, w: int, ddof: int = 1) -> np.ndarray:
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
    return np.sqrt(wvar(x, w, ddof))


def wmin(x: np.ndarray, w: int) -> np.ndarray:
    v = _views(x, w)
    acc = v[0].astype(F64, copy=True)
    for k in range(1, w):
        np.minimum(acc, v[k], out=acc)
    return acc


def wmax(x: np.ndarray, w: int) -> np.ndarray:
    v = _views(x, w)
    acc = v[0].astype(F64, copy=True)
    for k in range(1, w):
        np.maximum(acc, v[k], out=acc)
    return acc


def wtsrank(x: np.ndarray, w: int) -> np.ndarray:
    """Fraction of the window <= the newest value (1/W .. 1). NaN if any NaN in the window."""
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
    va, vb = _views(a, w), _views(b, w)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ma, mb = wsum(a, w) / w, wsum(b, w) / w
        acc = (va[0] - ma) * (vb[0] - mb)
        for k in range(1, w):
            acc += (va[k] - ma) * (vb[k] - mb)
        return acc / (w - ddof)


def wcorr(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    c = wcov(a, b, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return c / (wstd(a, w) * wstd(b, w))


def wshift(x: np.ndarray, w: int) -> np.ndarray:
    """Value W-1 rows back (shift by W-1)."""
    return x[: x.shape[0] - w + 1].copy()


def ewm_alpha(halflife: float) -> float:
    return 1.0 - math.exp(-math.log(2.0) / halflife)


def ewm_step(x: np.ndarray, prev: np.ndarray, alpha: float) -> np.ndarray:
    """One EWMA step. NaN input carries the previous value; NaN previous starts at x."""
    upd = alpha * x + (1.0 - alpha) * prev
    out = np.where(np.isnan(prev), x, upd)
    return np.where(np.isnan(x), prev, out)


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
    """Ordinal rank 1..k within each group among valid entries (ties by asset order)."""
    out = np.full(v.shape[0], np.nan)
    valid, g = _valid_groups(v, mask, groups)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    vals = v[idx] if ascending else -v[idx]
    gi = g[idx]
    order = np.lexsort((idx, vals, gi))  # group, then value, then asset index
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
    out = np.full(v.shape[0], np.nan)
    valid, g = _valid_groups(v, mask, groups)
    if not valid.any():
        return out
    gv, x, _cnt, mean = _group_moments(v, valid, g)
    out[valid] = x - mean[gv]
    return out


def cs_zscore(v: np.ndarray, mask: np.ndarray | None, groups: np.ndarray | None) -> np.ndarray:
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
