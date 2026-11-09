"""Validation against the planted ground truth.

* :func:`factor_recovery`: daily cross-sectional OLS of engine-computed log returns on the
  true exposures (beta, styles, industry dummies) recovers the planted factor returns; the
  engine's rolling time-series beta recovers each stock's true beta.
* :func:`signal_ic`: information coefficient of the planted signal (risk-adjusted residual
  IC, the quantity the generator plants, plus raw and rank IC on total returns).
* :func:`bar_distributions`: return distributions of clock bars vs tick/volume/dollar bars
  (excess kurtosis, skew, Jarque-Bera).
* :func:`decile_backtest`: long top decile / short bottom decile of the signal, using the
  engine's ``top``/``bottom`` filters, next-day returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from factorflow.batch import BatchEngine, BatchResult
from factorflow.events import AdaptiveDaily, build_event_bars
from factorflow.sim.market import STYLES, AssetSim, SimMarket
from factorflow.sim.pipelines import beta_window_for, daily_pipeline


# ====================================================================== stats helpers
def moments(x: np.ndarray) -> dict[str, float]:
    """n, skew, excess kurtosis and the Jarque-Bera test (chi2 with 2 dof: p = exp(-JB/2))."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return {"n": n, "skew": math.nan, "exkurt": math.nan, "jb": math.nan, "jb_p": math.nan}
    d = x - x.mean()
    m2 = float(np.mean(d * d))
    s = float(np.mean(d**3)) / m2**1.5
    k = float(np.mean(d**4)) / m2**2 - 3.0
    jb = n / 6.0 * (s * s + k * k / 4.0)
    return {"n": n, "skew": s, "exkurt": k, "jb": jb, "jb_p": math.exp(-jb / 2.0)}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return math.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    out[m] = np.argsort(np.argsort(x[m])).astype(float)
    return out


def _tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(x.size))) if x.size > 2 else math.nan


def run_daily(sim: AssetSim) -> BatchResult:
    """Run the daily research pipeline on one simulated book.

    Parameters
    ----------
    sim : AssetSim
        A book from :func:`simulate`.

    Returns
    -------
    BatchResult
        Outputs of :func:`factorflow.sim.pipelines.daily_pipeline`.
    """
    return BatchEngine().run(daily_pipeline(len(sim.assets), beta_window_for(sim.n_hist)), sim.daily)


# ====================================================================== factor recovery
@dataclass
class FactorRecovery:
    """How closely the engine's daily returns and betas match the planted factors.

    Notes
    -----
    ``factors`` maps a factor name to correlation, slope, bias in basis points,
    and volatility of the error. ``beta`` compares the engine's last rolling beta
    with the true beta. ``residuals`` is ``(T, N)``.
    """

    factors: dict[str, dict[str, float]]  # name -> corr / slope / bias / rmse_rel
    beta: dict[str, float]  # engine rolling beta vs true beta
    residuals: np.ndarray  # (T, N) regression residuals (NaN where not estimated)
    factor_hat: dict[str, np.ndarray]


def _design(sim: AssetSim) -> tuple[np.ndarray, list[str]]:
    tr = sim.truth
    g = int(tr.industry.max()) + 1
    dummies = (tr.industry[:, None] == np.arange(g)[None, :]).astype(float)
    X = np.column_stack([tr.beta, *[tr.exposures[k] for k in STYLES], dummies])
    names = ["market", *STYLES, *[f"ind{j}" for j in range(g)]]
    return X, names


def factor_recovery(sim: AssetSim, res: BatchResult | None = None, skip: int = 1) -> FactorRecovery:
    """Regress each day's engine log returns on the true exposures.

    Parameters
    ----------
    sim : AssetSim
        The book. Uses equities in :func:`validate`.
    res : BatchResult or None, optional
        A previous :func:`run_daily`. Computed if omitted.
    skip : int, optional
        Leading days dropped before the fit summary. Default is 1, so the
        first return, which has no lag, is skipped.

    Returns
    -------
    FactorRecovery
        Per-factor fit statistics and the residual panel.
    """
    res = res or run_daily(sim)
    R = res.values["logret"]
    X, names = _design(sim)
    t = R.shape[0]
    fh = np.full((t, X.shape[1]), np.nan)
    resid = np.full(R.shape, np.nan)
    for r in range(skip, t):
        y = R[r]
        m = np.isfinite(y)
        if m.sum() <= X.shape[1] + 2:
            continue
        coef, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        fh[r] = coef
        resid[r, m] = y[m] - X[m] @ coef
    tr = sim.truth
    truth = {"market": tr.factor_returns["market"], **{k: tr.factor_returns[k] for k in STYLES}}
    for j in range(tr.industry_returns.shape[1]):
        truth[f"ind{j}"] = tr.industry_returns[:, j]
    out: dict[str, dict[str, float]] = {}
    for j, k in enumerate(names):
        a, b = fh[skip:, j], truth[k][skip:]
        m = np.isfinite(a)
        slope = float(np.polyfit(b[m], a[m], 1)[0]) if m.sum() > 2 else math.nan
        out[k] = {
            "corr": _corr(a, b),
            "slope": slope,
            "bias_bp": 1e4 * float(np.mean(a[m] - b[m])),
            "true_vol_bp": 1e4 * float(np.std(b[m])),
            "err_vol_bp": 1e4 * float(np.std(a[m] - b[m])),
        }
    bh = res.values["beta"][-1]
    m = np.isfinite(bh)
    beta = {
        "corr": _corr(bh, tr.beta),
        "slope": float(np.polyfit(tr.beta[m], bh[m], 1)[0]) if m.sum() > 2 else math.nan,
        "mae": float(np.mean(np.abs(bh[m] - tr.beta[m] / tr.beta.mean()))),
        "names": int(m.sum()),
    }
    return FactorRecovery(out, beta, resid, {k: fh[:, j] for j, k in enumerate(names)})


# ====================================================================== IC
@dataclass
class ICResult:
    """Information coefficients of the planted signal.

    Notes
    -----
    ``resid_ic`` is the mean correlation of yesterday's signal with today's
    risk-adjusted residual, which is what the generator plants. ``raw_ic`` uses
    total next-day returns. ``rank_ic`` rank-correlates the signal with total returns.
    ``resid_ic_se`` is the standard error of the daily residual IC.
    """

    planted: float
    resid_ic: float  # mean risk-adjusted residual IC (what the generator plants)
    resid_ic_se: float
    raw_ic: float  # corr(signal_z, next-day total return)
    rank_ic: float
    days: int

    @property
    def z(self) -> float:
        """Return how many standard errors the residual IC sits from the planted value.

        Returns
        -------
        float
            ``(resid_ic - planted) / resid_ic_se``.
        """
        return (self.resid_ic - self.planted) / self.resid_ic_se


def signal_ic(sim: AssetSim, res: BatchResult | None = None, rec: FactorRecovery | None = None) -> ICResult:
    """Measure the planted signal's information coefficient on engine outputs.

    Parameters
    ----------
    sim : AssetSim
        The book.
    res : BatchResult or None, optional
        Daily pipeline result. Computed if omitted.
    rec : FactorRecovery or None, optional
        Residuals. Computed if omitted.

    Returns
    -------
    ICResult
        Residual, raw, and rank IC.
    """
    res = res or run_daily(sim)
    rec = rec or factor_recovery(sim, res)
    s = res.values["signal_z"]
    R = res.values["logret"]
    sig = sim.truth.idio_vol
    resid_ic, raw_ic, rank_ic = [], [], []
    for r in range(1, R.shape[0] - 1):
        e = rec.residuals[r + 1] / sig[r + 1]
        resid_ic.append(_corr(s[r], e))
        raw_ic.append(_corr(s[r], R[r + 1]))
        rank_ic.append(_corr(_rank(s[r]), _rank(R[r + 1])))
    a = np.array(resid_ic)
    a = a[np.isfinite(a)]
    return ICResult(
        sim.truth.ic,
        float(a.mean()),
        float(a.std(ddof=1) / math.sqrt(a.size)),
        float(np.nanmean(raw_ic)),
        float(np.nanmean(rank_ic)),
        int(a.size),
    )


# ====================================================================== backtest
@dataclass
class Backtest:
    """A decile long-short of the planted signal.

    Notes
    -----
    Returns are in basis points per day except ``sharpe`` (annualized, 252 days)
    and ``tstat`` (mean over standard error). ``expected_bp`` is the mean the
    planted kappa implies for a normal signal's top and bottom deciles.
    ``daily`` is the long-short return series, not in basis points.
    """

    days: int
    mean_bp: float  # mean daily long-short return
    vol_bp: float
    sharpe: float  # annualised
    tstat: float
    hit_rate: float
    expected_bp: float  # implied by the planted kappa
    long_n: float
    short_n: float
    daily: np.ndarray


def decile_backtest(sim: AssetSim, res: BatchResult | None = None) -> Backtest:
    """Equal-weight long ``long`` / short ``short`` (engine filters) at close t, return to t + 1."""
    res = res or run_daily(sim)
    close = sim.daily.columns["close"]
    fwd = np.full(close.shape, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    L, S = res.values["long"], res.values["short"]
    ls = []
    for r in range(1, close.shape[0] - 1):
        lo, sh = L[r] & np.isfinite(fwd[r]), S[r] & np.isfinite(fwd[r])
        if lo.any() and sh.any():
            ls.append(fwd[r, lo].mean() - fwd[r, sh].mean())
    x = np.array(ls)
    # E[LS] = kappa * E[sigma] * (E[s | top decile] - E[s | bottom decile]); for a normal
    # signal the decile means are +-phi(z_0.9) / 0.1 = +-1.755
    sig = float(np.mean(sim.truth.idio_vol))
    expected = sim.truth.kappa * sig * 2 * 1.755
    return Backtest(
        int(x.size),
        1e4 * float(x.mean()),
        1e4 * float(x.std(ddof=1)),
        float(x.mean() / x.std(ddof=1) * math.sqrt(252)),
        _tstat(x),
        float(np.mean(x > 0)),
        1e4 * expected,
        float(np.median(L.sum(axis=1)[L.any(axis=1)])),
        float(np.median(S.sum(axis=1)[S.any(axis=1)])),
        x,
    )


# ====================================================================== bar distributions
def _session_returns(close: np.ndarray, end: np.ndarray, day_of) -> np.ndarray:
    """Return log returns between consecutive bars in the same session.

    Overnight gaps are excluded: a return is kept only when both bars close
    on the same session day.
    """
    if close.size < 2:
        return np.empty(0)
    r = np.diff(np.log(close))
    same = day_of(end[1:]) == day_of(end[:-1])
    return r[same]


def bar_distributions(
    sim: AssetSim,
    clock_freqs: tuple[str, ...] = ("1min", "5min"),
    bars_per_day: int = 78,
    min_trades_per_day: float = 200.0,
) -> dict[str, dict[str, float]]:
    """Pooled per-asset-standardised intraday returns of clock bars vs event bars.

    Event bars use an adaptive threshold of ``bars_per_day`` per day (initialised from the
    history's mean daily measure, point-in-time), so they are comparable with 5-minute bars
    (78 per regular session). Illiquid names (< ``min_trades_per_day``) are excluded.
    """
    d = sim.daily.columns
    h = sim.n_hist
    n_hist_ticks = d["n_ticks"][:h].mean(axis=0) if h else np.full(len(sim.assets), np.inf)
    names = np.flatnonzero(n_hist_ticks >= min_trades_per_day)
    closes = sim.closes

    def session(ts: np.ndarray) -> np.ndarray:
        return np.searchsorted(closes, ts, side="left")

    out: dict[str, dict[str, float]] = {}
    per_asset_k: dict[str, list[float]] = {}
    pooled: dict[str, list[np.ndarray]] = {}
    for f in clock_freqs:
        p, _ = sim.time_bars(f)
        c = p.columns["close"]
        for j in names:
            r = _session_returns(c[:, j], p.times, session)
            r = r[np.isfinite(r)]
            if r.size > 10 and r.std() > 0:
                pooled.setdefault(f, []).append(r / r.std())
                per_asset_k.setdefault(f, []).append(moments(r)["exkurt"])
    tr = sim.trades
    measure_col = {"tick": "n_ticks", "volume": "volume", "dollar": "dollar"}
    for kind, col in measure_col.items():
        for j in names:
            m = tr.asset == j
            init = float(d[col][:h, j].mean()) / bars_per_day if h else 1.0
            bars, _ = build_event_bars(
                tr.ts[m],
                tr.price[m],
                tr.size[m],
                kind,
                AdaptiveDaily(bars_per_day, lookback_days=5, initial=init),
                "close",
                day_key=lambda ts: int(np.searchsorted(closes, ts, side="left")),
            )
            if len(bars) < 12:
                continue
            r = _session_returns(bars["close"], bars["end"], session)
            r = r[np.isfinite(r)]
            if r.size > 10 and r.std() > 0:
                pooled.setdefault(kind, []).append(r / r.std())
                per_asset_k.setdefault(kind, []).append(moments(r)["exkurt"])
    for k, rs in pooled.items():
        x = np.concatenate(rs)
        mo = moments(x)
        mo["median_asset_exkurt"] = float(np.median(per_asset_k[k]))
        mo["assets"] = len(rs)
        mo["bars_per_asset_day"] = x.size / len(rs) / max(len(sim.intraday), 1)
        out[k] = mo
    return out


# ====================================================================== everything
@dataclass
class Validation:
    """Factor recovery, signal IC, the decile backtest, and bar-shape statistics."""

    recovery: FactorRecovery
    ic: ICResult
    backtest: Backtest
    bars: dict[str, dict[str, float]]
    crypto_bars: dict[str, dict[str, float]] | None

    def report(self) -> str:
        """Format the validation as a text report.

        Returns
        -------
        str
            One block for factors, one for the signal, and one per asset class for bar distributions.
        """
        L = ["Factor recovery (daily cross-sectional OLS on true exposures):"]
        L.append(
            f"  {'factor':<9} {'corr':>6} {'slope':>6} {'bias bp':>8} {'true vol bp':>11} {'err vol bp':>10}"
        )
        for k, v in self.recovery.factors.items():
            if k.startswith("ind") and k != "ind0":
                continue
            L.append(
                f"  {k:<9} {v['corr']:6.3f} {v['slope']:6.3f} {v['bias_bp']:8.2f} "
                f"{v['true_vol_bp']:11.1f} {v['err_vol_bp']:10.1f}"
            )
        inds = [v for k, v in self.recovery.factors.items() if k.startswith("ind")]
        L.append(
            f"  industries ({len(inds)}): mean corr {np.mean([v['corr'] for v in inds]):.3f}, "
            f"mean slope {np.mean([v['slope'] for v in inds]):.3f}"
        )
        b = self.recovery.beta
        L.append(
            f"  rolling beta vs true beta ({b['names']} names): corr {b['corr']:.3f}, "
            f"slope {b['slope']:.3f}, MAE {b['mae']:.3f}"
        )
        ic = self.ic
        L.append("Planted signal:")
        L.append(
            f"  risk-adjusted residual IC {ic.resid_ic:.4f} ± {ic.resid_ic_se:.4f} "
            f"(planted {ic.planted:.4f}, "
            f"z = {ic.z:+.2f}) over {ic.days} days;  raw IC {ic.raw_ic:.4f}, rank IC {ic.rank_ic:.4f}"
        )
        bt = self.backtest
        L.append(
            f"  decile long-short ({bt.long_n:.0f} long / {bt.short_n:.0f} short): "
            f"mean {bt.mean_bp:.1f} bp/day "
            f"(expected ~{bt.expected_bp:.1f}), vol {bt.vol_bp:.0f} bp, Sharpe {bt.sharpe:.2f}, "
            f"t {bt.tstat:.2f}, hit {bt.hit_rate:.1%}"
        )
        for title, bars in (("Equity", self.bars), ("Crypto", self.crypto_bars)):
            if not bars:
                continue
            L.append(f"{title} intraday return distributions (pooled, standardised per asset):")
            L.append(
                f"  {'bars':<7} {'n':>8} {'bars/day':>8} {'skew':>6} {'ex.kurt':>8} "
                f"{'med kurt':>8} {'JB':>10}"
            )
            for k, v in bars.items():
                L.append(
                    f"  {k:<7} {v['n']:>8} {v['bars_per_asset_day']:8.1f} {v['skew']:6.2f} "
                    f"{v['exkurt']:8.2f} "
                    f"{v['median_asset_exkurt']:8.2f} {v['jb']:10.0f}"
                )
        return "\n".join(L)


def validate(mkt: SimMarket) -> Validation:
    """Run every check on a simulated market.

    Parameters
    ----------
    mkt : SimMarket
        From :func:`factorflow.sim.market.simulate`.

    Returns
    -------
    Validation
        Equity results, plus 24/7 bar distributions when the market has a crypto book.
    """
    eq = mkt.equities
    res = run_daily(eq)
    rec = factor_recovery(eq, res)
    ic = signal_ic(eq, res, rec)
    bt = decile_backtest(eq, res)
    bars = bar_distributions(eq)
    cb = None
    if mkt.crypto is not None:
        cb = bar_distributions(mkt.crypto, ("1min", "5min"), bars_per_day=288, min_trades_per_day=0)
    return Validation(rec, ic, bt, bars, cb)
