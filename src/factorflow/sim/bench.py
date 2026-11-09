"""Replay throughput and latency on simulated minute data (``python -m factorflow.sim bench``)."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass

import numpy as np

from factorflow.graph import compile_pipeline
from factorflow.grid import NS, BarGrid
from factorflow.live import LiveEngine
from factorflow.sim.market import SimConfig, simulate
from factorflow.sim.pipelines import minute_pipeline, wide_pipeline
from factorflow.sim.replay import Extras, ReplayStats, replay


@dataclass
class BenchRow:
    """One benchmark measurement.

    Notes
    -----
    ``names`` is the equity count. ``nodes`` is the compiled graph size.
    ``stats`` comes from :meth:`ReplayStats.summary`.
    """

    names: int
    pipeline: str
    nodes: int
    mode: str
    stats: dict[str, float]


def run_bench(
    names: tuple[int, ...] = (100, 500),
    minutes: int = 30,
    seed: int = 7,
    pipelines: tuple[str, ...] = ("demo", "wide100"),
) -> list[BenchRow]:
    """Replay the first ``minutes`` of a session for each universe size.

    The window is the busiest part of the intraday U-shape. The engine is a
    minute-grid :class:`~factorflow.live.LiveEngine` warmed on the previous
    session.

    Modes are ``ingest`` (ticks and seals, no revisions), ``revise@1s`` (a
    revision every second of market time), and ``revise@tick`` (a revision
    after every trade, first 5 minutes only).
    """
    rows: list[BenchRow] = []
    grid = BarGrid("XNYS", "1min")
    for n in names:
        cfg = SimConfig(seed=seed, history_days=30, intraday_days=2, crypto=None).scaled(n=n)
        eq = simulate(cfg).equities
        panel, _ = eq.time_bars("1min")
        first_day_bars = int(np.searchsorted(panel.times, eq.closes[eq.intraday[0]], side="right"))
        hist = panel.head(first_day_bars)
        extras = Extras(panel.times, {k: panel.columns[k] for k in ("signal", "industry")})
        t_open = int(eq.opens[eq.intraday[1]])
        start = int(np.searchsorted(eq.trades.ts, t_open, side="left"))
        stop = int(np.searchsorted(eq.trades.ts, t_open + minutes * 60 * NS, side="left"))
        stop5 = int(np.searchsorted(eq.trades.ts, t_open + 5 * 60 * NS, side="left"))
        for pname in pipelines:
            pipe = minute_pipeline() if pname == "demo" else wide_pipeline(100)
            g = compile_pipeline(pipe)
            for mode, kw, hi in (
                ("ingest", {"revise_every_ns": None}, stop),
                ("revise@1s", {"revise_every_ns": NS}, stop),
                ("revise@tick", {"revise_every_ns": None, "revise_every_ticks": 1}, stop5),
            ):
                live = LiveEngine(g, hist, grid)
                st = ReplayStats()
                replay(live, eq.trades, start=start, stop=hi, extras=extras, stats=st, **kw)  # type: ignore[arg-type]
                rows.append(BenchRow(n, pname, len(g.order), mode, st.summary()))
    return rows


def format_bench(rows: list[BenchRow], minutes: int) -> str:
    """Format benchmark rows as a text table.

    Parameters
    ----------
    rows : list of BenchRow
        Measurements from :func:`run_bench`.
    minutes : int
        How many minutes of the session were replayed, except the per-tick mode, which uses 5.

    Returns
    -------
    str
        A header plus one line per row. Latencies are milliseconds.
    """
    L = [
        f"factorflow replay benchmark: XNYS 1-minute grid, first {minutes} min of a session "
        f"(revise@tick: first 5 min)",
        f"python {platform.python_version()}  numpy {np.__version__}  "
        f"{platform.processor() or platform.machine()}  cpus={os.cpu_count()}",
        f"  {'names':>5} {'pipeline':<8} {'nodes':>5} {'mode':<11} {'ticks':>8} {'ticks/s':>9} "
        f"{'x realtime':>10} {'tick us':>7} {'revise p50/p95 ms':>18} {'seal p50/p95 ms':>16}",
    ]
    for r in rows:
        s = r.stats
        rev = f"{s['revise_ms_p50']:.2f}/{s['revise_ms_p95']:.2f}" if s["revises"] > 1 else "-"
        L.append(
            f"  {r.names:>5} {r.pipeline:<8} {r.nodes:>5} {r.mode:<11} {int(s['ticks']):>8} "
            f"{s['ticks_per_s']:>9,.0f} {s['speedup_vs_realtime']:>10,.0f} {s['tick_us']:>7.2f} "
            f"{rev:>18} {s['seal_ms_p50']:.2f}/{s['seal_ms_p95']:.2f}"
        )
    return "\n".join(L)


def main(names: tuple[int, ...], minutes: int, seed: int) -> None:
    """Run the benchmark and print the table.

    Parameters
    ----------
    names : tuple of int
        Equity universe sizes.
    minutes : int
        Minutes of the session to replay.
    seed : int
        SimConfig seed.
    """
    t0 = time.perf_counter()
    rows = run_bench(names, minutes, seed)
    print(format_bench(rows, minutes))
    print(f"(total {time.perf_counter() - t0:.0f} s)")
