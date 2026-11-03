"""Incremental executor: one open bar that ticks revise, sealed rows that never change.

State per node is a small ring buffer holding the last ``lookback - 1`` sealed rows plus one
*open slot*. ``revise()`` writes the open bar's inputs into the open slots and recomputes the
graph's last row in place, touching no sealed state, so revising is a pure function of
(sealed history, open bar). ``seal()`` makes the open slot permanent and opens the next bar.

Only dirty columns are recomputed for column-wise terms (elementwise and time-series), and a
node whose output did not change stops propagation (exact cutoff, NaN-aware). Cross-sectional
and row-only terms recompute the full row when any input column changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorflow.bars import FIELDS, BarState
from factorflow.batch import BatchEngine
from factorflow.graph import Graph, Pipeline, compile_pipeline
from factorflow.grid import BarGrid
from factorflow.panel import Panel
from factorflow.terms import Term, _Input

BAR_INPUTS = (*FIELDS, "has_trade")


class RowBuffer:
    """Sealed tail (``keep`` rows) + one open slot, contiguous for zero-copy windows."""

    __slots__ = ("cap", "data", "keep", "pos")

    def __init__(self, keep: int, n: int, dtype, missing, history: np.ndarray | None):
        self.keep = keep
        self.cap = max(2 * (keep + 1), keep + 64)
        self.data = np.full((self.cap, n), missing, dtype=dtype)
        self.pos = keep  # index of the open slot
        if history is not None and keep:
            tail = history[-keep:]
            self.data[keep - tail.shape[0] : keep] = tail

    def window(self, w: int) -> np.ndarray:
        return self.data[self.pos - w + 1 : self.pos + 1]

    @property
    def open(self) -> np.ndarray:
        return self.data[self.pos]

    def seal(self) -> None:
        self.pos += 1
        if self.pos == self.cap:
            k = self.keep
            self.data[:k] = self.data[self.pos - k : self.pos]
            self.pos = k
        self.data[self.pos] = self.data[self.pos - 1]  # open slot starts as a copy (overwritten)


@dataclass
class Snapshot:
    ts: int  # bar end (label), ns UTC
    is_open: bool  # True = provisional (open bar), False = sealed
    values: dict[str, np.ndarray]
    screen: np.ndarray | None
    changed: frozenset[str]  # outputs whose values changed since the previous snapshot
    assets: tuple[str, ...] = field(default=())

    def frame(self) -> pl.DataFrame:
        keep = self.screen if self.screen is not None else np.ones(len(self.assets), dtype=bool)
        idx = np.flatnonzero(keep)
        data = {
            "ts": pl.Series([self.ts] * idx.size, dtype=pl.Int64).cast(pl.Datetime("ns", "UTC")),
            "asset": [self.assets[i] for i in idx],
            "is_open": [self.is_open] * idx.size,
        }
        for k, v in self.values.items():
            data[k] = v[idx]
        return pl.DataFrame(data)


Observer = Callable[[Snapshot], None]


def _same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "f":
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


class LiveEngine:
    """Run a pipeline incrementally on top of a sealed history.

    With a ``grid``, ``tick()`` routes trades by timestamp and bars seal automatically once
    ``ts >= end + grace`` (``advance(now)`` does the same from a timer). Without a grid the
    caller drives the clock with ``seal(ts)``; that is how event bars (tick/volume/dollar)
    drive a per-asset pipeline.
    """

    def __init__(
        self,
        pipeline: Pipeline | Graph,
        history: Panel,
        grid: BarGrid | None = None,
        observers: Sequence[Observer] = (),
        partial_threshold: float = 0.0,
    ):
        self.graph = pipeline if isinstance(pipeline, Graph) else compile_pipeline(pipeline)
        self.assets = history.assets
        self.asset_index = {a: i for i, a in enumerate(self.assets)}
        self.n = n = len(self.assets)
        self.grid = grid
        self.observers: list[Observer] = list(observers)
        self.partial_threshold = partial_threshold
        g = self.graph

        warm = BatchEngine().run(g, history)
        self.bufs: dict[Term, RowBuffer] = {}
        for t in g.order:
            keep = g.lookback[t] - 1
            self.bufs[t] = RowBuffer(keep, n, t.dtype, t.missing, warm.nodes[t])
        self.prev: dict[Term, np.ndarray | None] = {
            t: warm.recursive_state.get(t) for t in g.order if t.recursive
        }
        self._consumers_window = g.lookback

        # open-row inputs
        self.bar = BarState(n)
        last_close = (
            history.columns["close"][-1].astype(np.float64)
            if "close" in history.columns and len(history.times)
            else np.full(n, np.nan)
        )
        self.bar.start(last_close)
        self.extra: dict[str, np.ndarray] = {}
        for t in g.inputs:
            name = t.name  # type: ignore[attr-defined]
            if name not in BAR_INPUTS:
                col = history.columns.get(name)
                if col is not None and len(history.times):
                    self.extra[name] = np.asarray(col[-1], dtype=t.dtype).copy()
                else:
                    self.extra[name] = np.full(n, t.missing, dtype=t.dtype)
        self._dirty = np.zeros(n, dtype=bool)
        self._fresh = True
        self._last_out: dict[str, np.ndarray] = {}
        self.late_ticks = 0
        self._pending: list[tuple[int, int, float, float]] = []

        self.last_sealed_ts = int(history.times[-1]) if len(history.times) else None
        self.open_start: int | None = None
        self.open_end: int | None = None
        if grid is not None:
            ref = self.last_sealed_ts if self.last_sealed_ts is not None else 0
            s, e = grid.bars_after(ref, 1)
            self.open_start, self.open_end = int(s[0]), int(e[0])

    # ------------------------------------------------------------------ inputs
    def _j(self, asset: str | int) -> int:
        return asset if isinstance(asset, (int, np.integer)) else self.asset_index[asset]  # type: ignore[return-value]

    def tick(self, asset: str | int, ts: int, price: float, size: float = 0.0) -> None:
        """A trade. With a grid, seals any bars that are now past ``end + grace`` first."""
        j = self._j(asset)
        if self.grid is not None:
            self.advance(ts)
            assert self.open_end is not None
            if ts >= self.open_end:  # inside the grace period: belongs to a later bar
                self._pending.append((j, ts, price, size))
                return
            if self.last_sealed_ts is not None and ts < self.last_sealed_ts:
                self.late_ticks += 1
                return
        self.bar.tick(j, price, size)
        self._dirty[j] = True

    def set_bar(
        self,
        asset: str | int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        dollar: float | None = None,
        n_ticks: float | None = None,
    ) -> None:
        j = self._j(asset)
        self.bar.set_bar(j, open, high, low, close, volume, dollar, n_ticks)
        self._dirty[j] = True

    def set_open(self, column: str, values: np.ndarray | Mapping[str | int, float]) -> None:
        """Set a non-bar input column for the open bar (full row array or {asset: value})."""
        f = self.bar.f[column] if column in BAR_INPUTS else self.extra[column]
        if isinstance(values, Mapping):
            for a, v in values.items():
                j = self._j(a)
                f[j] = v
                self._dirty[j] = True
        else:
            arr = np.asarray(values, dtype=f.dtype)
            self._dirty |= ~_same(arr, f)
            f[:] = arr

    def set_open_bar_fields(self, fields: Mapping[str, np.ndarray]) -> None:
        for k, v in fields.items():
            self.set_open(k, v)

    # ------------------------------------------------------------------ compute
    def _input_row(self, name: str) -> np.ndarray:
        if name in BAR_INPUTS:
            return self.bar.row(name)
        return self.extra[name]

    def _recompute(self) -> dict[Term, np.ndarray]:
        g, n = self.graph, self.n
        fresh = self._fresh
        changed: dict[Term, np.ndarray] = {}
        all_cols = np.ones(n, dtype=bool)
        for t in g.order:
            buf = self.bufs[t]
            if isinstance(t, _Input):
                buf.open[:] = self._input_row(t.name)  # type: ignore[attr-defined]
                changed[t] = all_cols if fresh else self._dirty
                continue
            if fresh:
                mask = all_cols
            else:
                mask = changed[t.inputs[0]]
                for i in t.inputs[1:]:
                    mask = mask | changed[i]
                if not mask.any():
                    changed[t] = mask
                    continue
            if t.rowwise:
                rows = [self.bufs[i].window(1) for i in t.inputs]
                new = t.compute_row(rows, self.prev.get(t))
                cols = None
            else:
                w = t.window
                k = int(mask.sum())
                if t.columnwise and not fresh and k < self.partial_threshold * n:
                    cols = np.flatnonzero(mask)
                    blocks = [self.bufs[i].window(w)[:, cols] for i in t.inputs]
                else:
                    cols = None
                    blocks = [self.bufs[i].window(w) for i in t.inputs]
                new = t.compute(blocks)[0]
            out = buf.open
            if fresh:
                out[:] = new
                changed[t] = all_cols
            elif cols is None:
                diff = ~_same(new, out)
                out[:] = new
                changed[t] = diff
            else:
                old = out[cols]
                d = ~_same(new, old)
                out[cols] = new
                m = np.zeros(n, dtype=bool)
                m[cols[d]] = True
                changed[t] = m
        self._dirty[:] = False
        self._fresh = False
        return changed

    def _snapshot(self, changed: dict[Term, np.ndarray], is_open: bool, ts: int | None) -> Snapshot:
        g = self.graph
        vals = {k: self.bufs[t].open.copy() for k, t in g.outputs.items()}
        names = frozenset(k for k, t in g.outputs.items() if changed[t].any())
        screen = self.bufs[g.screen].open.copy() if g.screen is not None else None
        stamp = ts if ts is not None else (self.open_end if self.open_end is not None else -1)
        return Snapshot(int(stamp), is_open, vals, screen, names, self.assets)

    def revise(self) -> Snapshot:
        """Recompute the open bar's row (no sealed state changes). Notifies observers on change."""
        changed = self._recompute()
        snap = self._snapshot(changed, True, None)
        if snap.changed:
            for ob in self.observers:
                ob(snap)
        return snap

    def seal(self, ts: int | None = None) -> Snapshot:
        """Make the open bar permanent and open the next one (missing bars carry the close)."""
        changed = self._recompute()
        if ts is None:
            if self.open_end is None:
                raise ValueError("seal(ts) needs a timestamp when the engine has no grid")
            ts = self.open_end
        snap = self._snapshot(changed, False, ts)
        for t in self.graph.order:
            if t.recursive:
                self.prev[t] = self.bufs[t].open.copy()
            self.bufs[t].seal()
        self.last_sealed_ts = int(ts)
        self.bar.start(self.bar.f["close"].copy())
        self._fresh = True
        if self.grid is not None:
            s, e = self.grid.bars_after(int(ts), 1)
            self.open_start, self.open_end = int(s[0]), int(e[0])
        for ob in self.observers:
            ob(snap)
        if self._pending:
            pend, self._pending = self._pending, []
            for j, t_, p, q in pend:
                self.tick(j, t_, p, q)
        return snap

    def advance(self, now: int) -> list[Snapshot]:
        """Seal every bar whose ``end + grace <= now`` (empty bars are forward-filled)."""
        out: list[Snapshot] = []
        if self.grid is None:
            return out
        while self.open_end is not None and now >= self.open_end + self.grid.grace_ns:
            out.append(self.seal())
        return out
