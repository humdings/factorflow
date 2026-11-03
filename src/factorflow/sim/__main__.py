"""``python -m factorflow.sim {demo,validate,bench}``."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from factorflow.grid import NS, BarGrid
from factorflow.live import LiveEngine, Snapshot
from factorflow.sim.market import EXTRA_COLUMNS, SimConfig, simulate
from factorflow.sim.pipelines import beta_window_for, crypto_pipeline, daily_pipeline, minute_pipeline
from factorflow.sim.replay import Extras, replay, verify_replay


def _fmt_ts(ns: int) -> str:
    return str(np.datetime64(int(ns), "ns").astype("datetime64[s]")).replace("T", " ") + " UTC"


def _verify(mkt, check_every: int = 0) -> bool:
    """Replay every intraday tick through LiveEngine and compare sealed rows with batch.

    Revision cadence (market time): daily bars every 60 s, equity minute bars every 1 s,
    crypto minute bars every 5 s.
    """
    eq, cr = mkt.equities, mkt.crypto
    p, _ = eq.time_bars("1min")
    first = int(np.searchsorted(p.times, eq.closes[eq.intraday[0]], side="right"))
    runs = [
        (
            "equities, daily bars (ticks revise the open daily bar)",
            eq,
            daily_pipeline(len(eq.assets), beta_window_for(eq.n_hist)),
            eq.daily,
            eq.daily_grid(),
            eq.n_hist,
            EXTRA_COLUMNS,
            60,
        ),
        (
            "equities, 1-minute bars",
            eq,
            minute_pipeline(),
            p,
            BarGrid("XNYS", "1min"),
            first,
            ("signal", "industry"),
            1,
        ),
    ]
    if cr is not None:
        cp, _ = cr.time_bars("1min")
        first_c = int(np.searchsorted(cp.times, cr.closes[cr.intraday[0]], side="right"))
        runs.append(
            ("crypto 24/7, 1-minute bars", cr, crypto_pipeline(), cp, BarGrid("24/7", "1min"), first_c, (), 5)
        )
    ok = True
    for label, sim, pipe, panel, grid, h, extra, every_s in runs:
        t0 = time.perf_counter()
        v = verify_replay(
            pipe,
            panel,
            sim.trades,
            grid,
            h,
            extra_columns=extra,
            revise_every_ns=every_s * NS,
            check_open_every=check_every,
        )
        s = v.stats.summary()
        status = "OK " if v.ok else "FAIL"
        print(
            f"  [{status}] {label}: {v.sealed_rows} sealed rows == batch (bit for bit), "
            f"{v.open_checked} open revisions checked; {int(s['ticks']):,} ticks, {s['revises']} revisions "
            f"every {every_s} s, revise p50 {s['revise_ms_p50']:.2f} ms ({time.perf_counter() - t0:.1f} s)"
        )
        for m in v.mismatches[:5]:
            print("        ", m)
        ok &= v.ok
    return ok


def _live_view(mkt, speed: float, seconds: float, top: int, out=sys.stdout) -> None:
    """Replay the last session's ticks through a minute-grid engine, printing live values."""
    eq = mkt.equities
    p, _ = eq.time_bars("1min")
    last = eq.intraday[-1]
    h = int(np.searchsorted(p.times, eq.opens[last], side="right"))
    live = LiveEngine(minute_pipeline(), p.head(h), BarGrid("XNYS", "1min"))
    extras = Extras(p.times, {k: p.columns[k] for k in ("signal", "industry")})
    t_open = int(eq.opens[last])
    start = int(np.searchsorted(eq.trades.ts, t_open, side="left"))
    stop = int(np.searchsorted(eq.trades.ts, t_open + int(seconds * speed * NS), side="left"))
    tty = out.isatty()
    last_print = [0.0]
    n_sealed = [0]
    assets = eq.assets

    def show(snap: Snapshot, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - last_print[0] < 1.0:
            return
        last_print[0] = now
        v = snap.values
        order = [int(i) for i in np.argsort(v["rank"]) if np.isfinite(v["rank"][i])][:top]
        flag = "LIVE (open bar, provisional)" if snap.is_open else "SEALED (final)"
        lines = [
            f"bar ending {_fmt_ts(snap.ts)}  [{flag}]  sealed bars this run: {n_sealed[0]}  "
            f"changed: {', '.join(sorted(snap.changed)) or '-'}",
            f"  {'asset':<6} {'is_open':<7} {'close':>9} {'mom30':>8} {'vol30':>8} {'vwap_dev':>9} "
            f"{'score':>7} {'rank':>4}",
        ]
        for i in order:
            lines.append(
                f"  {assets[i]:<6} {snap.is_open!s:<7} {v['close'][i]:9.2f} {v['mom30'][i]:8.4f} "
                f"{v['vol30'][i]:8.5f} {v['vwap_dev'][i]:9.5f} {v['score'][i]:7.3f} {v['rank'][i]:4.0f}"
            )
        if tty:
            out.write("\x1b[2J\x1b[H")
        out.write("\n".join(lines) + "\n\n")
        out.flush()

    def on_seal(snap: Snapshot) -> None:
        n_sealed[0] += 1
        show(snap)

    st = replay(
        live, eq.trades, start=start, stop=stop, extras=extras, speed=speed, on_revise=show, on_seal=on_seal
    )
    snap = live.revise()
    show(snap, force=True)
    s = st.summary()
    print(
        f"replayed {int(s['ticks']):,} ticks ({st.market_s / 60:.1f} market minutes at {speed:g}x), "
        f"{len(st.seal_ms)} bars sealed; revise p50 {s['revise_ms_p50']:.2f} ms, "
        f"seal p50 {s['seal_ms_p50']:.2f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factorflow.sim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "validate", "bench"):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, default=7)
        if name != "bench":
            sp.add_argument("--names", type=int, default=100)
            sp.add_argument("--history-days", type=int, default=252)
            sp.add_argument("--intraday-days", type=int, default=3)
        if name == "demo":
            sp.add_argument("--speed", type=float, default=60.0, help="market seconds per wall second")
            sp.add_argument("--seconds", type=float, default=15.0, help="wall seconds of live replay")
            sp.add_argument("--top", type=int, default=10)
            sp.add_argument("--skip-validation", action="store_true")
        if name == "validate":
            sp.add_argument(
                "--check-open",
                type=int,
                default=0,
                help="also check every Nth open revision against batch (slow)",
            )
        if name == "bench":
            sp.add_argument("--names", type=int, nargs="+", default=[100, 500])
            sp.add_argument("--minutes", type=int, default=30)
    a = ap.parse_args(argv)

    if a.cmd == "bench":
        from factorflow.sim.bench import main as bench_main

        bench_main(tuple(a.names), a.minutes, a.seed)
        return 0

    from factorflow.sim.validate import validate

    cfg = SimConfig(seed=a.seed, history_days=a.history_days, intraday_days=a.intraday_days).scaled(n=a.names)
    t0 = time.perf_counter()
    mkt = simulate(cfg)
    print(f"== simulated market (seed {a.seed}, {time.perf_counter() - t0:.1f} s)")
    print(mkt.summary())
    ok = True
    if a.cmd == "validate" or not a.skip_validation:
        print("\n== live replay vs batch")
        ok = _verify(mkt, getattr(a, "check_open", 0))
        print("\n== validation against the planted truth")
        print(validate(mkt).report())
    if a.cmd == "demo":
        print(f"\n== live: last session replayed at {a.speed:g}x for {a.seconds:g} s (Ctrl-C to stop)")
        try:
            _live_view(mkt, a.speed, a.seconds, a.top)
        except KeyboardInterrupt:
            print("stopped")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
