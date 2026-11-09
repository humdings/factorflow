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
    """Sealed tail plus one open slot, stored so a window is a view.

    Parameters
    ----------
    keep : int
        How many sealed rows to retain. This is ``lookback - 1``: the rows
        before the open row that a consumer's window still needs.
    n : int
        Number of assets.
    dtype : numpy dtype
        Stored dtype. Factors are float64, filters bool, classifiers int64.
    missing
        Value used to fill slots that history did not cover. NaN, ``False``,
        or ``-1`` depending on the term.
    history : ndarray of shape (T, N) or None
        Sealed output to warm from. The last ``keep`` rows are copied in.
        ``None`` or a short history leaves the older slots at ``missing``,
        which is the warmup a batch run would also show.
    """

    __slots__ = ("cap", "data", "keep", "pos")

    def __init__(self, keep: int, n: int, dtype, missing, history: np.ndarray | None):
        """Allocate the ring and copy the sealed tail.

        Parameters
        ----------
        keep : int
            Sealed rows to retain.
        n : int
            Number of assets.
        dtype : numpy dtype
            Stored dtype.
        missing
            Fill value for slots history does not cover.
        history : ndarray or None
            Prior sealed output. Only the last ``keep`` rows are kept.
        """
        self.keep = keep
        self.cap = max(2 * (keep + 1), keep + 64)
        self.data = np.full((self.cap, n), missing, dtype=dtype)
        self.pos = keep  # index of the open slot
        if history is not None and keep:
            tail = history[-keep:]
            self.data[keep - tail.shape[0] : keep] = tail

    def window(self, w: int) -> np.ndarray:
        """Return the last ``w`` rows, ending at the open slot.

        Parameters
        ----------
        w : int
            Window length in bars, including the open row. The live engine
            passes a term's own ``window``, so the view is exactly what
            batch would pad out to the same length.

        Returns
        -------
        ndarray of shape (w, N)
            A view. Row ``-1`` is the open slot.
        """
        return self.data[self.pos - w + 1 : self.pos + 1]

    @property
    def open(self) -> np.ndarray:
        """The open slot.

        Returns
        -------
        ndarray of shape (N,)
            A view. Writes here are the provisional row.
        """
        return self.data[self.pos]

    def seal(self) -> None:
        """Commit the open slot and open the next one.

        The new open slot starts as a copy of the row just sealed, which is
        the forward-filled close for bar fields until a tick overwrites it.
        When the buffer is full, the last ``keep`` sealed rows slide to the
        front. Older rows are dropped; consumers only needed ``keep`` of them.
        """
        self.pos += 1
        if self.pos == self.cap:
            k = self.keep
            self.data[:k] = self.data[self.pos - k : self.pos]
            self.pos = k
        self.data[self.pos] = self.data[self.pos - 1]  # open slot starts as a copy (overwritten)


@dataclass
class Snapshot:
    """One pipeline row, either provisional or sealed.

    Parameters
    ----------
    ts : int
        Bar end, nanoseconds UTC. For an open clock bar this is the end the
        bar will have once it seals, not the timestamp of the last trade.
    is_open : bool
        ``True`` while the bar can still change. ``False`` after :meth:`LiveEngine.seal`.
    values : dict of str to ndarray
        Named outputs, each shape ``(N,)``. Copies, so later ticks do not
        mutate a snapshot the caller kept.
    screen : ndarray of bool, shape (N,), or None
        The screen on this row, or ``None`` if the pipeline has no screen.
    changed : frozenset of str
        Output names whose values differ from the previous snapshot. Empty
        when nothing a watcher cares about moved. NaN equals NaN for this test.
    assets : tuple of str, optional
        Asset names, column order of the arrays.
    """

    ts: int  # bar end (label), ns UTC
    is_open: bool  # True = provisional (open bar), False = sealed
    values: dict[str, np.ndarray]
    screen: np.ndarray | None
    changed: frozenset[str]  # outputs whose values changed since the previous snapshot
    assets: tuple[str, ...] = field(default=())

    def frame(self) -> pl.DataFrame:
        """Stack this row into a one-bar long frame.

        Returns
        -------
        polars.DataFrame
            Columns ``ts``, ``asset``, ``is_open``, and each output. Assets
            that fail the screen are omitted. With no screen, every asset is
            present.
        """
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
        """Warm from ``history`` and open the next bar.

        Parameters
        ----------
        pipeline : Pipeline or Graph
            What to run. Compiled once, here.
        history : Panel
            Sealed bars. Must include every input the pipeline reads, and
            should be at least as long as the longest window or the early
            live rows stay missing, the same way batch warmup is missing.
        grid : BarGrid or None, optional
            Clock that seals bars from timestamps. ``None`` means the caller
            seals with :meth:`seal` (event bars do this).
        observers : sequence of callable, optional
            Each is called with a :class:`Snapshot` when a revise changes an
            output, and on every seal. Default is none.
        partial_threshold : float, optional
            Fraction of assets. Column-wise terms recompute only the dirty
            columns when fewer than this fraction changed. ``0`` recomputes
            the whole row, which is faster at a few hundred names because
            the Python loop over lags dominates. Default is ``0``.

        Raises
        ------
        KeyError
            If ``history`` lacks a column the pipeline reads. Raised while
            the batch warm-up runs.
        """
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
        """Fold one trade into the open bar.

        Parameters
        ----------
        asset : str or int
            Asset name, or a column index.
        ts : int
            Trade time, nanoseconds UTC.
        price : float
            Trade price, in the caller's price units.
        size : float, optional
            Trade size. Default is 0, which still counts as a tick and moves
            the close. Volume does not increase.

        Notes
        -----
        With a grid, bars whose ``end + grace`` is at or before ``ts`` are
        sealed first. A trade stamped at or after the open bar's end is held
        and applied after that seal. A trade older than the last sealed bar
        is dropped and counted in :attr:`late_ticks`.
        """
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
        """Replace one asset's open OHLCV bar.

        Parameters
        ----------
        asset : str or int
            Asset name, or a column index.
        open, high, low, close : float
            The in-progress bar's prices.
        volume : float
            Volume so far, in the caller's size units.
        dollar : float or None, optional
            Dollar volume. ``None`` stores ``close * volume``.
        n_ticks : float or None, optional
            Trade count. ``None`` stores 1 when ``volume > 0``, else 0.

        Notes
        -----
        Does not seal. Call :meth:`revise` to recompute factors. The asset
        is marked dirty.
        """
        j = self._j(asset)
        self.bar.set_bar(j, open, high, low, close, volume, dollar, n_ticks)
        self._dirty[j] = True

    def set_open(self, column: str, values: np.ndarray | Mapping[str | int, float]) -> None:
        """Set a column of the open bar that is not implied by trades.

        Parameters
        ----------
        column : str
            An input the pipeline reads. Bar fields (``open``, ``close``, ...)
            and extra columns (a sector code, a signal) are both accepted.
            The column must already be an input; this does not add outputs.
        values : ndarray of shape (N,) or mapping
            A full row, or ``{asset: value}`` for a subset. Dtype follows the
            stored column (float64 for factors, int64 for classifiers).

        Notes
        -----
        Assets whose value changed are marked dirty. Call :meth:`revise` afterwards.
        """
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
        """Set several open-bar columns at once.

        Parameters
        ----------
        fields : mapping of str to ndarray
            Each value is a full row, passed to :meth:`set_open`.
        """
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
                # ``compute`` returns ``T_in - window + 1`` rows. This call
                # passes exactly ``window`` rows (sealed tail + open slot), so
                # the result has one row: the same arithmetic as the last row
                # of a batch run that treats this open bar as final.
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
        """Recompute the open bar.

        Returns
        -------
        Snapshot
            ``is_open`` is ``True``. Sealed rows are not modified, so calling
            this twice with no new input returns the same values. Observers
            run only when an output changed.

        Notes
        -----
        A revision equals a batch run on the sealed history plus this open
        bar appended as a final row.
        """
        changed = self._recompute()
        snap = self._snapshot(changed, True, None)
        if snap.changed:
            for ob in self.observers:
                ob(snap)
        return snap

    def seal(self, ts: int | None = None) -> Snapshot:
        """Commit the open bar and start the next one.

        Parameters
        ----------
        ts : int or None, optional
            Bar end, nanoseconds UTC. ``None`` uses the grid's open-bar end.
            Required when the engine was built without a grid.

        Returns
        -------
        Snapshot
            The row just sealed, with ``is_open`` False. Observers always run.

        Raises
        ------
        ValueError
            If ``ts`` is omitted and there is no grid.

        Notes
        -----
        Assets with no trade keep the previous close and get zero volume.
        Trades that arrived during the grace period, stamped into the next
        bar, are applied after the seal.
        """
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
        """Seal every bar whose grace has elapsed.

        Parameters
        ----------
        now : int
            Current time, nanoseconds UTC. A bar seals when
            ``now >= end + grace``. Empty bars are sealed too, carrying the
            close forward.

        Returns
        -------
        list of Snapshot
            One sealed snapshot per bar closed by this call, in time order.
            Empty if nothing was due, or if the engine has no grid.
        """
        out: list[Snapshot] = []
        if self.grid is None:
            return out
        while self.open_end is not None and now >= self.open_end + self.grid.grace_ns:
            out.append(self.seal())
        return out
