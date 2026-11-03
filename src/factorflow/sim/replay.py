"""Replay a trade stream through :class:`~factorflow.LiveEngine` and check it against batch.

``replay`` streams trades in timestamp order. Before each trade it calls ``advance(ts)``
(sealing every bar whose ``end + grace`` has passed, forward-filling empty bars), then
``tick``. The open bar is revised on a market-time cadence (``revise_every_ns``, like a
dashboard refresh) and/or every ``revise_every_ticks`` trades. With ``speed`` set, the
driver sleeps so that market time runs ``speed`` times faster than the wall clock (1.0 =
real time); without it, it runs as fast as possible.

Non-bar inputs (signal, classifiers, style exposures) come from ``extras``: a
``(times, {column: (T, N) array})`` pair aligned to bar ends; whenever a new bar opens, its
row is written with ``set_open``, exactly as a feed handler would.

``verify_replay`` runs the batch engine on the same bars and asserts that every sealed live
row is bit-for-bit equal to the batch row (and every sampled open revision equals a batch run
on sealed history + the open bar treated as final).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from factorflow.bars import FIELDS, Trades
from factorflow.batch import BatchEngine
from factorflow.events import EventBarBuilder, EventBars, EventPipeline, build_event_bars
from factorflow.graph import Graph, Pipeline, compile_pipeline
from factorflow.grid import NS, BarGrid
from factorflow.live import LiveEngine, Snapshot
from factorflow.panel import Panel


@dataclass
class ReplayStats:
    n_ticks: int = 0
    wall_s: float = 0.0
    market_s: float = 0.0
    tick_s: float = 0.0  # time inside tick() (bar accumulation + routing)
    revise_ms: list[float] = field(default_factory=list)
    seal_ms: list[float] = field(default_factory=list)
    sealed: list[Snapshot] = field(default_factory=list)
    open_checked: int = 0

    @property
    def ticks_per_s(self) -> float:
        return self.n_ticks / self.wall_s if self.wall_s > 0 else float("nan")

    def _q(self, xs: list[float], q: float) -> float:
        return float(np.quantile(xs, q)) if xs else float("nan")

    def summary(self) -> dict[str, float]:
        return {
            "ticks": self.n_ticks,
            "wall_s": self.wall_s,
            "ticks_per_s": self.ticks_per_s,
            "speedup_vs_realtime": self.market_s / self.wall_s if self.wall_s else float("nan"),
            "tick_us": 1e6 * self.tick_s / self.n_ticks if self.n_ticks else float("nan"),
            "revises": len(self.revise_ms),
            "revise_ms_p50": self._q(self.revise_ms, 0.5),
            "revise_ms_p95": self._q(self.revise_ms, 0.95),
            "revise_ms_p99": self._q(self.revise_ms, 0.99),
            "seals": len(self.seal_ms),
            "seal_ms_p50": self._q(self.seal_ms, 0.5),
            "seal_ms_p95": self._q(self.seal_ms, 0.95),
        }


class Extras:
    """Row lookup for non-bar input columns: the row of the first bar end >= ``open_end``."""

    def __init__(self, times: np.ndarray, columns: Mapping[str, np.ndarray]):
        self.times = np.asarray(times, np.int64)
        self.columns = dict(columns)

    def row(self, open_end: int) -> dict[str, np.ndarray]:
        r = min(int(np.searchsorted(self.times, open_end, side="left")), len(self.times) - 1)
        return {k: v[r] for k, v in self.columns.items()}


def replay(
    engine: LiveEngine,
    trades: Trades,
    *,
    start: int = 0,
    stop: int | None = None,
    revise_every_ns: int | None = NS,
    revise_every_ticks: int | None = None,
    extras: Extras | None = None,
    speed: float | None = None,
    until: int | None = None,
    on_revise: Callable[[Snapshot], None] | None = None,
    on_seal: Callable[[Snapshot], None] | None = None,
    check_open: Callable[[LiveEngine, Snapshot], None] | None = None,
    check_open_every: int = 0,
    stats: ReplayStats | None = None,
) -> ReplayStats:
    """Stream ``trades[start:stop]`` through ``engine`` (which must have a grid).

    ``until`` (ns) finally advances the clock to seal every bar ending by then.
    ``check_open(engine, snap)`` is called on every ``check_open_every``-th revision.
    """
    if engine.grid is None:
        raise ValueError("replay needs a LiveEngine with a BarGrid")
    st = stats or ReplayStats()
    stop = len(trades) if stop is None else stop
    a_, t_, p_, q_ = trades.asset, trades.ts, trades.price, trades.size
    pc = time.perf_counter
    wall0 = pc()
    t_first = int(t_[start]) if stop > start else 0
    last_open = None

    def sealed_now(snaps: list[Snapshot], dt: float) -> None:
        if not snaps:
            return
        per = 1e3 * dt / len(snaps)
        for s in snaps:
            st.seal_ms.append(per)
            st.sealed.append(s)
            if on_seal is not None:
                on_seal(s)

    def open_bar_inputs() -> None:
        nonlocal last_open
        if extras is not None and engine.open_end != last_open:
            for k, v in extras.row(int(engine.open_end)).items():  # type: ignore[arg-type]
                if k in engine.extra:  # columns the pipeline does not read are skipped
                    engine.set_open(k, v)
        last_open = engine.open_end

    open_bar_inputs()
    next_revise = (t_first // revise_every_ns + 1) * revise_every_ns if revise_every_ns else None
    since = 0
    n_rev = 0
    for i in range(start, stop):
        ts = int(t_[i])
        if speed:
            # pace market time; keep sealing bars on the clock while no trades arrive
            while (lag := (ts - t_first) / NS / speed - (pc() - wall0)) > 0:
                time.sleep(min(lag, 0.05))
                now = t_first + int((pc() - wall0) * speed * NS)
                t0 = pc()
                snaps = engine.advance(min(now, ts))
                sealed_now(snaps, pc() - t0)
                if snaps:
                    open_bar_inputs()
        # market-time revisions due before this trade
        if next_revise is not None and ts >= next_revise:
            if since:
                n_rev += 1
                _timed_revise(engine, st, on_revise, check_open, check_open_every, n_rev)
                since = 0
            next_revise = (ts // revise_every_ns + 1) * revise_every_ns  # type: ignore[operator]
        t0 = pc()
        snaps = engine.advance(ts)
        sealed_now(snaps, pc() - t0)
        if snaps:
            open_bar_inputs()
        t0 = pc()
        engine.tick(int(a_[i]), ts, float(p_[i]), float(q_[i]))
        st.tick_s += pc() - t0
        since += 1
        if revise_every_ticks and since >= revise_every_ticks:
            n_rev += 1
            _timed_revise(engine, st, on_revise, check_open, check_open_every, n_rev)
            since = 0
    if since:
        n_rev += 1
        _timed_revise(engine, st, on_revise, check_open, check_open_every, n_rev)
    if until is not None:
        t0 = pc()
        snaps = engine.advance(until)
        sealed_now(snaps, pc() - t0)
    st.n_ticks += stop - start
    st.wall_s += pc() - wall0
    if stop > start:
        st.market_s += (int(t_[stop - 1]) - t_first) / NS
    return st


def _timed_revise(engine, st, on_revise, check_open, every, n) -> None:
    t0 = time.perf_counter()
    snap = engine.revise()
    st.revise_ms.append(1e3 * (time.perf_counter() - t0))
    if on_revise is not None:
        on_revise(snap)
    if check_open is not None and every and n % every == 0:
        check_open(engine, snap)
        st.open_checked += 1


# ====================================================================== verification
def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


@dataclass
class VerifyResult:
    grid: str
    sealed_rows: int
    open_checked: int
    mismatches: list[str]
    stats: ReplayStats

    @property
    def ok(self) -> bool:
        return not self.mismatches and self.sealed_rows > 0


def verify_replay(
    pipeline: Pipeline | Graph,
    full: Panel,
    trades: Trades,
    grid: BarGrid,
    history_rows: int,
    *,
    extra_columns: tuple[str, ...] = (),
    revise_every_ns: int | None = NS,
    revise_every_ticks: int | None = None,
    check_open_every: int = 0,
    partial_threshold: float = 0.0,
    label: str = "",
) -> VerifyResult:
    """Live replay of the trades after ``full.head(history_rows)`` vs batch on ``full``.

    ``full`` must be the batch bars of the same trades on ``grid`` (``build_time_bars``),
    optionally preceded by history rows, plus any ``extra_columns``.
    """
    g = pipeline if isinstance(pipeline, Graph) else compile_pipeline(pipeline)
    batch = BatchEngine().run(g, full)
    hist = full.head(history_rows)
    live = LiveEngine(g, hist, grid, partial_threshold=partial_threshold)
    extras = Extras(full.times, {k: full.columns[k] for k in extra_columns}) if extra_columns else None
    start = int(np.searchsorted(trades.ts, int(full.times[history_rows - 1]), side="left"))
    mism: list[str] = []

    def check_open(eng: LiveEngine, snap: Snapshot) -> None:
        n_sealed = history_rows + len(stats.sealed)
        bar_cols = (*FIELDS, "has_trade")
        row = {}
        for k in full.columns:
            if k in bar_cols:
                row[k] = eng.bar.row(k)
            elif k in eng.extra:
                row[k] = eng.extra[k]
            else:  # not a pipeline input; any value will do
                row[k] = full.columns[k][min(n_sealed, len(full.times) - 1)]
        ref = BatchEngine().run(g, full.head(n_sealed).with_row(row, int(eng.open_end)))  # type: ignore[arg-type]
        for k, v in snap.values.items():
            if not _equal(v, ref.values[k][n_sealed]):
                mism.append(f"open row {n_sealed} column {k}")

    stats = ReplayStats()
    replay(
        live,
        trades,
        start=start,
        revise_every_ns=revise_every_ns,
        revise_every_ticks=revise_every_ticks,
        extras=extras,
        until=int(full.times[-1]) + grid.grace_ns,
        check_open=check_open if check_open_every else None,
        check_open_every=check_open_every,
        stats=stats,
    )
    sealed = stats.sealed
    expect = len(full.times) - history_rows
    if len(sealed) != expect:
        mism.append(f"sealed {len(sealed)} bars, expected {expect}")
    for j, snap in enumerate(sealed[:expect]):
        r = history_rows + j
        if snap.ts != int(full.times[r]):
            mism.append(f"sealed bar {j} stamped {snap.ts}, batch row {r} is {int(full.times[r])}")
            break
        for k, v in snap.values.items():
            if not _equal(v, batch.values[k][r]):
                mism.append(f"sealed row {r} column {k}")
        if batch.screen is not None and snap.screen is not None and not _equal(snap.screen, batch.screen[r]):
            mism.append(f"sealed row {r} screen")
    return VerifyResult(label or repr(grid), len(sealed), stats.open_checked, mism, stats)


def verify_event_pipeline(
    pipeline: Pipeline | Graph,
    trades: Trades,
    asset: int,
    builder_factory: Callable[[], EventBarBuilder],
    history_frac: float = 0.3,
    label: str = "",
) -> VerifyResult:
    """One asset's event bars (tick/volume/dollar): live EventPipeline vs batch."""
    g = pipeline if isinstance(pipeline, Graph) else compile_pipeline(pipeline)
    m = trades.asset == asset
    ts, px, sz = trades.ts[m], trades.price[m], trades.size[m]
    b = builder_factory()
    bars, _ = build_event_bars(ts, px, sz, b.kind, b.rule, b.overshoot, b.day_key)
    batch = BatchEngine().run(g, bars.panel())
    cut = int(len(ts) * history_frac)
    hb = builder_factory()
    hist_bars = EventBars.from_bars(hb.update_many(ts[:cut].tolist(), px[:cut].tolist(), sz[:cut].tolist()))
    # resume the same builder state: replay the history trades into it, then go live
    ep = EventPipeline(g, hb, history=hist_bars)
    stats = ReplayStats()
    t0 = time.perf_counter()
    sealed: list[Snapshot] = []
    for t, p, q in zip(ts[cut:].tolist(), px[cut:].tolist(), sz[cut:].tolist(), strict=True):
        for s in ep.update(t, p, q):
            if not s.is_open:
                sealed.append(s)
    stats.wall_s = time.perf_counter() - t0
    stats.n_ticks = len(ts) - cut
    mism: list[str] = []
    h = len(hist_bars)
    if h + len(sealed) != len(bars):
        mism.append(f"live sealed {h + len(sealed)} bars, batch has {len(bars)}")
    for j, s in enumerate(sealed):
        r = h + j
        if r >= len(bars):
            break
        for k, v in s.values.items():
            if not _equal(v, batch.values[k][r]):
                mism.append(f"event row {r} column {k}")
    return VerifyResult(label or f"event[{asset}]", len(sealed), 0, mism, stats)
