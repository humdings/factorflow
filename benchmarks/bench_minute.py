"""Benchmark: 500 names × 100 factors on a 1-minute grid; cost of updating the open (last) row.

    uv run python benchmarks/bench_minute.py [--names 500] [--factors 100] [--dirty 25]

Reports, per call (median / p95 over many iterations):
  * revise, sparse  - after ticks on ``--dirty`` names (the common live case)
  * revise, full    - first revision of a fresh bar (every node, every column)
  * seal            - seal the bar (includes a final revise) + open the next one
  * batch last row  - for scale: re-running the whole pipeline in batch on the history
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time

import numpy as np

import factorflow as ff


def make_pipeline(n_factors: int, sector: ff.Classifier) -> ff.Pipeline:
    """Build a wide minute pipeline of ``n_factors`` terms.

    Parameters
    ----------
    n_factors : int
        How many output columns to generate. Windows cycle through a fixed set
        of lengths, in bars.
    sector : Classifier
        Industry codes used by the grouped terms.

    Returns
    -------
    Pipeline
        Screened to names with positive 30-bar average volume.
    """
    c, v, h, lo = ff.BarData.close, ff.BarData.volume, ff.BarData.high, ff.BarData.low
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
        lambda w: c.pct_change(w).zscore(mask=liquid, groupby=sector),
        lambda w: r.rolling_std(w).rank(mask=liquid),
        lambda w: (r.rolling_mean(w) / r.rolling_std(w)).demean(groupby=sector),
        lambda w: r.rolling_beta(r.demean(), w),
    ]
    cols = {}
    i = 0
    while len(cols) < n_factors:
        g = gens[i % len(gens)]
        w = windows[(i // len(gens)) % len(windows)]
        cols[f"f{len(cols):03d}"] = g(w)
        i += 1
    return ff.Pipeline(cols, screen=liquid)


def pct(xs: list[float], q: float) -> float:
    """Return an empirical quantile of a small sample.

    Parameters
    ----------
    xs : list of float
        Observations, in seconds.
    q : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    float
        ``xs`` sorted, at index ``q * len(xs)`` clipped to the last element.
    """
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    """Parse arguments, time revise and seal, and print the table.

    Synthetic closes are a seeded random walk. Nothing is read from disk.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=500)
    ap.add_argument("--factors", type=int, default=100)
    ap.add_argument("--history", type=int, default=390)
    ap.add_argument("--dirty", type=int, default=25, help="names ticked between revisions")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--partial",
        type=float,
        default=None,
        help="LiveEngine partial_threshold (fraction of dirty names below which only dirty "
        "columns are recomputed); default: engine default",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    n, t = args.names, args.history
    grid = ff.BarGrid("XNYS", "1min")
    s, e = grid.bars("2026-10-26", "2026-10-30")
    s, e = s[: t + 10], e[: t + 10]
    assets = tuple(f"S{i:03d}" for i in range(n))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, (t, n)), axis=0))
    vol = rng.integers(100, 10_000, (t, n)).astype(float)
    cols = {
        "close": close,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "volume": vol,
        "dollar": vol * close,
        "n_ticks": np.ones((t, n)),
        "has_trade": np.ones((t, n), bool),
        "sector": np.tile(np.arange(n) % 11, (t, 1)),
    }
    hist = ff.Panel(e[:t], assets, cols)
    pipe = make_pipeline(args.factors, ff.Column("sector", ff.Classifier))
    g = ff.compile_pipeline(pipe)

    t0 = time.perf_counter()
    ff.BatchEngine().run(g, hist)
    batch_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    kw = {} if args.partial is None else {"partial_threshold": args.partial}
    live = ff.LiveEngine(g, hist, grid, **kw)
    warm_s = time.perf_counter() - t0

    sparse, full, seal = [], [], []
    ts = int(live.open_start)  # type: ignore[arg-type]
    for _ in range(args.iters):
        t0 = time.perf_counter()
        live.revise()  # fresh bar: full recompute
        full.append(time.perf_counter() - t0)
        for _ in range(5):
            for j in rng.choice(n, args.dirty, replace=False):
                ts += 1
                live.tick(int(j), ts, float(close[-1, j] * (1 + rng.normal(0, 1e-3))), 100.0)
            t0 = time.perf_counter()
            live.revise()
            sparse.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        live.seal()
        seal.append(time.perf_counter() - t0)
        ts = int(live.open_start)  # type: ignore[arg-type]

    nodes = len(g.order)
    print(f"factorflow benchmark  names={n} factors={args.factors} graph_nodes={nodes} history={t} bars")
    print(
        f"python {platform.python_version()}  numpy {np.__version__}  "
        f"{platform.processor() or platform.machine()}  cpus={os.cpu_count()}"
    )
    for name, xs in (
        (f"revise, sparse ({args.dirty} dirty)", sparse),
        ("revise, full (fresh bar)", full),
        ("seal (+ final revise)", seal),
    ):
        print(
            f"  {name:<28} median {statistics.median(xs) * 1e3:7.2f} ms   p95 {pct(xs, 0.95) * 1e3:7.2f} ms"
        )
    print(
        f"  {'batch, whole history':<28} {batch_s * 1e3:7.0f} ms   (LiveEngine warm-up {warm_s * 1e3:.0f} ms)"
    )


if __name__ == "__main__":
    main()
