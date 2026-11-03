"""A synthetic market with known ground truth.

Daily returns follow a planted factor model (log returns)::

    R[t, i] = beta[i] * f_mkt[t] + sum_k X[i, k] * f_k[t] + f_ind[t, g(i)]
              + kappa * sigma[t, i] * s[t - 1, i] + sigma[t, i] * eps[t, i]

* ``f_mkt`` and ``sigma`` (idiosyncratic vol) follow GARCH(1, 1) with Student-t shocks, so
  returns are fat-tailed with volatility clustering;
* ``X`` are style exposures (``size`` = z-scored log market cap, ``value``, ``quality``),
  ``g`` is an industry classifier, ``beta`` the market beta;
* ``s`` is a planted alpha signal (AR(1), standardised cross-sectionally each day). It is
  published as the ``signal`` column for day ``t`` and predicts day ``t + 1``. Its
  risk-adjusted residual IC, ``corr(s[t], (R[t + 1] - factor part) / sigma[t + 1])``, is
  ``ic`` by construction (``kappa = ic / sqrt(1 - ic**2)``).

Intraday, each (asset, day) gets a trade stream: Poisson arrivals with a U-shaped (equities)
or diurnal (crypto) intensity times a persistent random activity multiplier, lognormal trade
sizes, and a price path that is a Brownian bridge **in volume time** pinned to the day's
close (so ``Var(dlog p) ∝ traded volume``, Clark's subordination model). Volume time makes
volume and dollar bars close to Gaussian while clock bars inherit the U-shape and activity
bursts and come out fat-tailed. Trades print at the mid ± a half spread (bid/ask bounce),
rounded to the tick. A few names are illiquid (missing minute bars), some (asset, day) pairs
have a trading halt (no trades, then a jump at the reopen), and the XNYS calendar supplies
holidays and early closes (2026-11-27 closes at 13:00 ET).

Everything is a pure function of :class:`SimConfig` (seeded ``numpy`` generators).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from factorflow.bars import Trades, build_time_bars
from factorflow.grid import NS, BarGrid
from factorflow.panel import Panel

MIN = 60 * NS
STYLES = ("size", "value", "quality")


# ====================================================================== configuration
@dataclass(frozen=True)
class EquitySpec:
    n: int = 100
    n_industries: int = 8
    style_vol: tuple[float, ...] = (0.004, 0.003, 0.0025)  # daily vol of each style factor return
    market_vol: float = 0.010  # long-run daily vol of the market factor
    industry_vol: float = 0.008
    beta_mean: float = 1.0
    beta_sd: float = 0.3
    idio_vol: float = 0.016  # median long-run daily idiosyncratic vol
    garch: tuple[float, float] = (0.08, 0.90)  # (alpha, beta); persistence alpha + beta < 1
    t_df: float = 5.0  # Student-t degrees of freedom of every shock
    ic: float = 0.03  # risk-adjusted residual IC of the planted signal
    signal_ar: float = 0.8  # day-to-day persistence of the signal
    trades_per_day: float = 1500.0  # median liquid name
    illiquid_frac: float = 0.05  # smallest names trade rarely -> missing minute bars
    illiquid_trades_per_day: float = 25.0
    halt_prob: float = 0.02  # per (asset, intraday day)
    halt_minutes: tuple[int, int] = (15, 60)
    overnight_frac: float = 0.2  # share of daily variance realised overnight (the open gap)
    half_spread_bps: float = 2.0
    tick: float = 0.01
    mean_trade_size: float = 200.0
    size_sigma: float = 1.0  # lognormal sigma of trade sizes
    activity_sd: float = 0.5  # sd of the log activity multiplier (intraday bursts)
    activity_ar: float = 0.97  # per-minute persistence of the activity multiplier


@dataclass(frozen=True)
class CryptoSpec:
    names: tuple[str, ...] = ("BTC", "ETH", "SOL")
    start_price: tuple[float, ...] = (95_000.0, 3_500.0, 180.0)
    beta: tuple[float, ...] = (1.0, 1.2, 1.5)  # to the BTC factor
    idio_vol: tuple[float, ...] = (0.004, 0.018, 0.03)
    market_vol: float = 0.028  # daily vol of the BTC factor
    garch: tuple[float, float] = (0.10, 0.88)
    t_df: float = 4.0
    trades_per_day: tuple[float, ...] = (20_000.0, 12_000.0, 8_000.0)
    mean_trade_size: tuple[float, ...] = (0.05, 0.8, 15.0)  # in coins
    size_sigma: float = 1.2
    half_spread_bps: float = 0.5
    tick: float = 0.01
    activity_sd: float = 0.5
    activity_ar: float = 0.98


@dataclass(frozen=True)
class SimConfig:
    seed: int = 7
    end: str = "2026-11-30"  # last session (inclusive); the default window spans the 11-27 early close
    history_days: int = 252  # daily bars only
    intraday_days: int = 3  # the last sessions, which also get trades (ticks)
    equities: EquitySpec = field(default_factory=EquitySpec)
    crypto: CryptoSpec | None = field(default_factory=CryptoSpec)

    def scaled(self, n: int | None = None, **kw) -> SimConfig:
        """A copy with ``n`` equities (and any other top-level fields replaced)."""
        cfg = replace(self, **kw)
        return replace(cfg, equities=replace(cfg.equities, n=n)) if n is not None else cfg


# ====================================================================== outputs
@dataclass
class Truth:
    """The planted ground truth (daily rows aligned with ``AssetSim.daily``)."""

    beta: np.ndarray  # (N,)
    exposures: dict[str, np.ndarray]  # style -> (N,)
    industry: np.ndarray  # (N,) int codes (empty for crypto)
    factor_returns: dict[str, np.ndarray]  # "market" and each style -> (T,)
    industry_returns: np.ndarray  # (T, G)
    idio_vol: np.ndarray  # (T, N) conditional idiosyncratic vol sigma[t, i]
    total_vol: np.ndarray  # (T, N) conditional total daily vol (used to scale the intraday path)
    signal: np.ndarray  # (T, N) s[t]; predicts R[t + 1]
    log_returns: np.ndarray  # (T, N) planted close-to-close log returns
    ic: float
    kappa: float


@dataclass
class AssetSim:
    """One asset class: daily bars for every session, trades for the last ``intraday_days``."""

    name: str
    calendar: str
    assets: tuple[str, ...]
    sessions: np.ndarray  # (T,) datetime64[D] session dates
    opens: np.ndarray  # (T,) session open, ns UTC
    closes: np.ndarray  # (T,) session close, ns UTC
    daily: Panel  # (T, N) daily bars (+ signal / industry / style columns for equities)
    trades: Trades  # trades of the intraday sessions, sorted by ts
    intraday: np.ndarray  # row indices (into ``daily``) of the sessions that have trades
    halts: list[tuple[int, int, int]]  # (asset index, halt start ns, halt end ns)
    truth: Truth

    @property
    def n_hist(self) -> int:
        """Daily rows before the first intraday session (the rows not built from trades)."""
        return int(self.intraday[0]) if self.intraday.size else len(self.sessions)

    def daily_grid(self) -> BarGrid:
        return BarGrid(self.calendar, "1d")

    def extras(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Non-bar input columns (signal, classifier, styles) for daily rows ``rows``."""
        return {k: v[rows] for k, v in self.daily.columns.items() if k in EXTRA_COLUMNS}

    def time_bars(self, freq: str) -> tuple[Panel, np.ndarray]:
        """Clock bars of the intraday sessions on a ``freq`` grid, with the extra columns of
        each bar's session. Returns ``(panel, session_row_of_each_bar)``."""
        grid = BarGrid(self.calendar, freq)
        d0, d1 = str(self.sessions[self.intraday[0]]), str(self.sessions[self.intraday[-1]])
        s, e = grid.bars(d0, d1)
        prev = self.daily.columns["close"][self.n_hist - 1] if self.n_hist else None
        p = build_time_bars(self.trades, self.assets, s, e, prev_close=prev)
        day = np.searchsorted(self.closes, e, side="left")  # session whose close >= bar end
        for k, v in self.extras(day).items():
            p.columns[k] = v
        return p, day


EXTRA_COLUMNS = ("signal", "industry", *STYLES)


@dataclass
class SimMarket:
    config: SimConfig
    equities: AssetSim
    crypto: AssetSim | None

    def summary(self) -> str:
        lines = []
        for a in (self.equities, self.crypto):
            if a is None:
                continue
            tr = a.trades
            d = a.daily.columns
            idays = [str(a.sessions[i]) for i in a.intraday]
            lines.append(
                f"{a.name:<9} {len(a.assets):>4} assets  {len(a.sessions)} daily bars "
                f"({a.sessions[0]} .. {a.sessions[-1]}, calendar {a.calendar})  "
                f"{len(tr):,} trades on {', '.join(idays)}"
            )
            last = d["close"][-1]
            lines.append(
                f"          last close: median {np.nanmedian(last):,.2f}, "
                f"range {np.nanmin(last):,.2f} .. {np.nanmax(last):,.2f};  halts: {len(a.halts)}"
            )
        e = self.equities
        lines.append(
            f"planted: {len(STYLES)} styles {STYLES}, {e.truth.industry.max() + 1} industries, "
            f"market beta ~ N({self.config.equities.beta_mean}, {self.config.equities.beta_sd}), "
            f"signal IC {e.truth.ic:.3f} (kappa {e.truth.kappa:.4f}), t({self.config.equities.t_df}) shocks"
        )
        return "\n".join(lines)


# ====================================================================== daily model
def _t_shocks(rng: np.random.Generator, df: float, shape) -> np.ndarray:
    """Student-t shocks scaled to unit variance."""
    return rng.standard_t(df, size=shape) * math.sqrt((df - 2.0) / df)


def _garch(rng: np.random.Generator, uncond: np.ndarray, t: int, ab: tuple[float, float], df: float):
    """GARCH(1, 1) paths: returns (shocks z (T, N), conditional vols (T, N)), x = vol * z."""
    a, b = ab
    uncond = np.atleast_1d(np.asarray(uncond, dtype=float))
    omega = uncond**2 * (1.0 - a - b)
    z = _t_shocks(rng, df, (t, uncond.size))
    vol = np.empty((t, uncond.size))
    var = uncond**2
    for r in range(t):
        vol[r] = np.sqrt(var)
        x = vol[r] * z[r]
        var = omega + a * x * x + b * var
    return z, vol


def _calendar_sessions(calendar: str, end: str, count: int):
    grid = BarGrid(calendar, "1d")
    end_d = np.datetime64(end, "D")
    span = int(count * 1.6) + 10
    s, e = grid.bars(str(end_d - np.timedelta64(span, "D")), end)
    s, e = s[-count:], e[-count:]
    if len(e) < count:
        raise ValueError("calendar too short for the requested history")
    dates = (e - 1).astype("datetime64[ns]").astype("datetime64[D]")
    return dates, s, e


def _equity_daily(rng: np.random.Generator, spec: EquitySpec, t: int):
    n, g = spec.n, spec.n_industries
    log_cap = rng.normal(23.0, 1.5, n)
    size = (log_cap - log_cap.mean()) / log_cap.std()
    value = rng.standard_normal(n)
    quality = 0.3 * size + math.sqrt(1 - 0.09) * rng.standard_normal(n)  # mildly correlated styles
    exposures = {"size": size, "value": value, "quality": (quality - quality.mean()) / quality.std()}
    industry = rng.permutation(np.arange(n) % g).astype(np.int64)
    beta = np.clip(rng.normal(spec.beta_mean, spec.beta_sd, n), 0.2, 2.5)
    idio_long = spec.idio_vol * np.exp(-0.25 * size + rng.normal(0, 0.2, n))

    z_m, vol_m = _garch(rng, np.array([spec.market_vol]), t, spec.garch, spec.t_df)
    f_mkt = (vol_m * z_m)[:, 0]
    styles = {k: spec.style_vol[i] * _t_shocks(rng, spec.t_df, t) for i, k in enumerate(STYLES)}
    f_ind = spec.industry_vol * _t_shocks(rng, spec.t_df, (t, g))
    eps, sig = _garch(rng, idio_long, t, spec.garch, spec.t_df)

    # planted signal: AR(1), standardised cross-sectionally every day
    s = np.empty((t, n))
    raw = rng.standard_normal(n)
    for r in range(t):
        raw = spec.signal_ar * raw + math.sqrt(1 - spec.signal_ar**2) * rng.standard_normal(n)
        s[r] = (raw - raw.mean()) / raw.std()
    kappa = spec.ic / math.sqrt(1.0 - spec.ic**2)
    alpha = np.zeros((t, n))
    alpha[1:] = kappa * sig[1:] * s[:-1]

    common = beta[None, :] * f_mkt[:, None] + f_ind[:, industry]
    for k in STYLES:
        common += exposures[k][None, :] * styles[k][:, None]
    R = common + alpha + sig * eps
    fvar = (
        beta[None, :] ** 2 * vol_m**2
        + spec.industry_vol**2
        + np.sum([(exposures[k] * spec.style_vol[i]) ** 2 for i, k in enumerate(STYLES)], axis=0)[None, :]
    )
    total_vol = np.sqrt(fvar + sig**2)
    truth = Truth(
        beta,
        exposures,
        industry,
        {"market": f_mkt, **styles},
        f_ind,
        sig,
        total_vol,
        s,
        R,
        spec.ic,
        kappa,
    )
    return truth, log_cap


def _crypto_daily(rng: np.random.Generator, spec: CryptoSpec, t: int):
    n = len(spec.names)
    beta = np.asarray(spec.beta, dtype=float)
    z_m, vol_m = _garch(rng, np.array([spec.market_vol]), t, spec.garch, spec.t_df)
    f = (vol_m * z_m)[:, 0]
    eps, sig = _garch(rng, np.asarray(spec.idio_vol), t, spec.garch, spec.t_df)
    R = beta[None, :] * f[:, None] + sig * eps
    total = np.sqrt(beta[None, :] ** 2 * vol_m**2 + sig**2)
    truth = Truth(
        beta,
        {},
        np.zeros(0, np.int64),
        {"market": f},
        np.zeros((t, 0)),
        sig,
        total,
        np.zeros((t, n)),
        R,
        0.0,
        0.0,
    )
    return truth


# ====================================================================== intraday model
def _profile(kind: str, open_ns: int, n_min: int) -> np.ndarray:
    """Relative trade intensity per minute (mean 1)."""
    u = (np.arange(n_min) + 0.5) / max(n_min, 1)
    if kind == "equity":  # U shape: busy open, quiet lunch, busy close
        w = 0.55 + 2.2 * np.exp(-u / 0.05) + 1.3 * np.exp(-(1 - u) / 0.07)
    else:  # 24/7: diurnal, peaking during US/EU overlap (~15:00 UTC)
        hours = ((open_ns // MIN + np.arange(n_min)) % 1440) / 60.0
        w = 1.0 + 0.5 * np.cos(2 * np.pi * (hours - 15.0) / 24.0)
    return w / w.mean()


def _activity(rng: np.random.Generator, n_min: int, sd: float, ar: float) -> np.ndarray:
    a = np.empty(n_min)
    x = rng.normal(0, sd)
    innov = rng.normal(0, sd * math.sqrt(1 - ar * ar), n_min)
    for m in range(n_min):
        x = ar * x + innov[m]
        a[m] = x
    return np.exp(a - 0.5 * sd * sd)


def _day_trades(
    rng: np.random.Generator,
    *,
    kind: str,
    open_ns: int,
    close_ns: int,
    n_expected: float,
    prev_close: float,
    target_log_ret: float,
    gap: float,
    day_vol: float,
    mean_size: float,
    size_sigma: float,
    half_spread: float,
    tick: float,
    activity_sd: float,
    activity_ar: float,
    halt: tuple[int, int] | None,
    integer_sizes: bool,
    overnight_frac: float,
):
    """Trades of one asset in one session: (ts, price, size)."""
    n_min = int((close_ns - open_ns) // MIN)
    lam = _profile(kind, open_ns, n_min) * _activity(rng, n_min, activity_sd, activity_ar)
    lam *= n_expected / lam.sum()
    if halt is not None:
        h0, h1 = halt
        lam[h0:h1] = 0.0
    counts = rng.poisson(lam)
    k = int(counts.sum())
    if k == 0:
        return np.empty(0, np.int64), np.empty(0), np.empty(0)
    minute = np.repeat(np.arange(n_min), counts)
    ts = open_ns + minute * MIN + rng.integers(0, MIN, k)
    ts.sort()
    size = mean_size * np.exp(rng.normal(-0.5 * size_sigma**2, size_sigma, k))
    size = np.maximum(np.round(size), 1.0) if integer_sizes else np.maximum(np.round(size, 6), 1e-6)

    # variance weights: traded volume (volume time); the first trade after a halt carries the
    # volume the halted minutes would have seen (the reopening jump)
    w = size.copy()
    if halt is not None:
        h0, h1 = halt
        after = np.searchsorted(minute, h1, side="left")
        if after < k:
            w[after] += mean_size * float(n_expected * (_profile(kind, open_ns, n_min)[h0:h1].sum() / n_min))
    W = w.sum()
    intraday_var = day_vol**2 * (1.0 - overnight_frac)
    z = rng.standard_normal(k) * np.sqrt(intraday_var * w / W)
    target = target_log_ret - gap
    x = z + (w / W) * (target - z.sum())  # bridge: increments sum exactly to the target
    logp = math.log(prev_close) + gap + np.cumsum(x)
    side = rng.choice((-1.0, 1.0), k)
    price = np.exp(logp) * (1.0 + side * half_spread)
    price = np.maximum(np.round(price / tick) * tick, tick)
    price = np.round(price, 8)
    return ts.astype(np.int64), price, size


def _daily_bars_from_model(
    rng: np.random.Generator,
    closes: np.ndarray,
    gaps: np.ndarray,
    vol: np.ndarray,
    n_trades: np.ndarray,
    mean_size: np.ndarray,
    tick: float,
    first_prev: np.ndarray,
) -> dict[str, np.ndarray]:
    prev = np.vstack([first_prev[None, :], closes[:-1]])
    o = prev * np.exp(gaps)
    hi = np.maximum(o, closes) * np.exp(np.abs(rng.normal(0, 0.4, closes.shape)) * vol)
    lo = np.minimum(o, closes) * np.exp(-np.abs(rng.normal(0, 0.4, closes.shape)) * vol)

    def rnd(x):
        return np.round(np.maximum(np.round(x / tick) * tick, tick), 8)

    o, hi, lo, c = rnd(o), rnd(hi), rnd(lo), rnd(closes)
    hi = np.maximum.reduce([hi, o, c])
    lo = np.minimum.reduce([lo, o, c])
    volume = np.round(n_trades * mean_size[None, :], 6)
    return {
        "open": o,
        "high": hi,
        "low": lo,
        "close": c,
        "volume": volume,
        "dollar": volume * 0.5 * (o + c),
        "n_ticks": n_trades.astype(float),
        "has_trade": n_trades > 0,
    }


def _simulate_class(
    rng: np.random.Generator,
    *,
    name: str,
    calendar: str,
    kind: str,
    assets: tuple[str, ...],
    cfg: SimConfig,
    truth: Truth,
    start_price: np.ndarray,
    trades_per_day: np.ndarray,
    mean_size: np.ndarray,
    size_sigma: float,
    half_spread: float,
    tick: float,
    activity: tuple[float, float],
    overnight_frac: float,
    halt_prob: float,
    halt_minutes: tuple[int, int],
    extra: dict[str, np.ndarray],
) -> AssetSim:
    t_all = cfg.history_days + cfg.intraday_days
    dates, opens, closes_ns = _calendar_sessions(calendar, cfg.end, t_all)
    R = truth.log_returns
    n = len(assets)
    close_px = start_price[None, :] * np.exp(np.cumsum(R, axis=0))
    gaps = (
        overnight_frac * R
        + rng.normal(0, 1, R.shape)
        * truth.total_vol
        * math.sqrt(max(overnight_frac * (1 - overnight_frac), 0.0))
        if overnight_frac > 0
        else np.zeros_like(R)
    )
    # activity drives variance: trades per day scale with (vol / long-run vol)^2
    rel = truth.total_vol / truth.total_vol.mean(axis=0, keepdims=True)
    n_trades_exp = trades_per_day[None, :] * rel**2 * np.exp(rng.normal(-0.02, 0.2, R.shape))
    h = cfg.history_days
    n_trades_hist = rng.poisson(n_trades_exp[:h]).astype(float)
    bars = _daily_bars_from_model(
        rng, close_px[:h], gaps[:h], truth.total_vol[:h], n_trades_hist, mean_size, tick, start_price
    )

    ts_l, px_l, sz_l, as_l = [], [], [], []
    halts: list[tuple[int, int, int]] = []
    hist_close = bars["close"][-1] if h else start_price
    for r in range(h, t_all):
        o_ns, c_ns = int(opens[r]), int(closes_ns[r])
        n_min = (c_ns - o_ns) // MIN
        prev = close_px[r - 1] if r else start_price
        if r == h:
            prev = hist_close  # the last sealed daily close
        for j in range(n):
            halt = None
            if halt_prob > 0 and rng.random() < halt_prob and n_min > 120:
                dur = int(rng.integers(halt_minutes[0], halt_minutes[1] + 1))
                h0 = int(rng.integers(30, n_min - dur - 30))
                halt = (h0, h0 + dur)
                halts.append((j, o_ns + h0 * MIN, o_ns + (h0 + dur) * MIN))
            ts, px, sz = _day_trades(
                rng,
                kind=kind,
                open_ns=o_ns,
                close_ns=c_ns,
                n_expected=float(n_trades_exp[r, j]),
                prev_close=float(prev[j]),
                target_log_ret=float(math.log(close_px[r, j] / prev[j])),
                gap=float(gaps[r, j]),
                day_vol=float(truth.total_vol[r, j]),
                mean_size=float(mean_size[j]),
                size_sigma=size_sigma,
                half_spread=half_spread,
                tick=tick,
                activity_sd=activity[0],
                activity_ar=activity[1],
                halt=halt,
                integer_sizes=kind == "equity",
                overnight_frac=overnight_frac,
            )
            ts_l.append(ts)
            px_l.append(px)
            sz_l.append(sz)
            as_l.append(np.full(ts.size, j, np.int64))
    ts = np.concatenate(ts_l) if ts_l else np.empty(0, np.int64)
    order = np.argsort(ts, kind="stable")
    trades = Trades(
        np.concatenate(as_l)[order] if as_l else np.empty(0, np.int64),
        ts[order],
        np.concatenate(px_l)[order] if px_l else np.empty(0),
        np.concatenate(sz_l)[order] if sz_l else np.empty(0),
    )
    # intraday daily rows are built from the trades (as a live feed would)
    intraday = np.arange(h, t_all)
    cols = dict(bars)
    if intraday.size:
        built = build_time_bars(trades, assets, opens[h:], closes_ns[h:], prev_close=hist_close)
        for k in cols:
            cols[k] = np.concatenate([cols[k], built.columns[k]], axis=0)
    for k, v in extra.items():
        cols[k] = v
    daily = Panel(closes_ns, assets, cols, opens)
    return AssetSim(name, calendar, assets, dates, opens, closes_ns, daily, trades, intraday, halts, truth)


def simulate(config: SimConfig | None = None) -> SimMarket:
    """Generate the market (deterministic for a given config)."""
    cfg = config or SimConfig()
    ss = np.random.SeedSequence(cfg.seed)
    r_eq_daily, r_eq_intra, r_cr_daily, r_cr_intra = (np.random.default_rng(s) for s in ss.spawn(4))
    t_all = cfg.history_days + cfg.intraday_days
    es = cfg.equities

    truth, log_cap = _equity_daily(r_eq_daily, es, t_all)
    n = es.n
    assets = tuple(f"EQ{i:03d}" for i in range(n))
    start_price = np.round(np.exp(r_eq_daily.normal(math.log(60.0), 0.8, n)), 2)
    tpd = es.trades_per_day * np.exp(0.5 * (log_cap - np.median(log_cap)))
    tpd = np.clip(tpd, 0.2 * es.trades_per_day, 6.0 * es.trades_per_day)
    n_ill = round(es.illiquid_frac * n)
    if n_ill:
        tpd[np.argsort(log_cap)[:n_ill]] = es.illiquid_trades_per_day
    extra = {
        "signal": truth.signal,
        "industry": np.tile(truth.industry, (t_all, 1)),
        **{k: np.tile(v, (t_all, 1)) for k, v in truth.exposures.items()},
    }
    equities = _simulate_class(
        r_eq_intra,
        name="equities",
        calendar="XNYS",
        kind="equity",
        assets=assets,
        cfg=cfg,
        truth=truth,
        start_price=start_price,
        trades_per_day=tpd,
        mean_size=np.full(n, es.mean_trade_size),
        size_sigma=es.size_sigma,
        half_spread=es.half_spread_bps * 1e-4,
        tick=es.tick,
        activity=(es.activity_sd, es.activity_ar),
        overnight_frac=es.overnight_frac,
        halt_prob=es.halt_prob,
        halt_minutes=es.halt_minutes,
        extra=extra,
    )
    crypto = None
    if cfg.crypto is not None:
        cs = cfg.crypto
        ctruth = _crypto_daily(r_cr_daily, cs, t_all)
        crypto = _simulate_class(
            r_cr_intra,
            name="crypto",
            calendar="24/7",
            kind="crypto",
            assets=tuple(cs.names),
            cfg=cfg,
            truth=ctruth,
            start_price=np.asarray(cs.start_price, dtype=float),
            trades_per_day=np.asarray(cs.trades_per_day, dtype=float),
            mean_size=np.asarray(cs.mean_trade_size, dtype=float),
            size_sigma=cs.size_sigma,
            half_spread=cs.half_spread_bps * 1e-4,
            tick=cs.tick,
            activity=(cs.activity_sd, cs.activity_ar),
            overnight_frac=0.0,
            halt_prob=0.0,
            halt_minutes=(0, 0),
            extra={},
        )
    return SimMarket(cfg, equities, crypto)
