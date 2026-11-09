"""Information-driven bars (López de Prado, *AFML* ch. 2): tick, volume and dollar bars.

An event bar closes when the running *measure* reaches a threshold:

=========  ===========================
kind       measure added per trade
=========  ===========================
``tick``   1
``volume`` size
``dollar`` price × size
=========  ===========================

Event bars are **per asset** (each asset has its own clock). Use them for time-series
factors through a per-asset pipeline (:class:`EventPipeline`), and use :func:`sample_asof` /
:class:`AsofSampler` to put event-bar values onto a common clock grid, as of the last
*sealed* event bar, before taking cross-sectional ranks.

Overshoot (the trade that crosses the threshold): ``"close"`` (default) puts the whole
trade in the bar it completes, so bars hold whole trades and ``measure >= threshold``;
``"split"`` cuts the trade so each bar holds exactly ``threshold`` and the remainder starts
the next bar(s). A very large trade can then fill several bars that share its timestamp.

Thresholds are :class:`Fixed` or :class:`AdaptiveDaily`. The adaptive threshold for a day is
``mean(measure per day over the previous lookback days) / target_bars_per_day``, using only
days that are complete before the day's first trade (point-in-time, no lookahead). A new
threshold takes effect at the first trade of a new day; the open bar is not reset at
day boundaries.

Batch and incremental construction run the same state machine (:class:`EventBarBuilder`),
so batch output equals incremental output by construction; the tests check it anyway on
random streams with random chunking.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from factorflow.graph import Graph, Pipeline
from factorflow.grid import NS
from factorflow.live import LiveEngine, Snapshot
from factorflow.panel import Panel

EVENT_FIELDS = ("start", "end", "open", "high", "low", "close", "volume", "dollar", "n_ticks", "threshold")
KINDS = ("tick", "volume", "dollar")
OVERSHOOT = ("close", "split")


def utc_day(ts: int) -> int:
    """Return the UTC day index of a timestamp.

    Parameters
    ----------
    ts : int
        Nanoseconds since the Unix epoch, UTC.

    Returns
    -------
    int
        ``ts // (86400 * 1e9)``. The default day key for adaptive thresholds.
    """
    return int(ts) // (86400 * NS)


@dataclass(frozen=True)
class Fixed:
    """A threshold that does not change.

    Parameters
    ----------
    value : float
        The threshold, in the bar's measure (ticks, size, or price times size). Must be positive.

    Raises
    ------
    ValueError
        If ``value`` is not positive. Raised from ``__post_init__``.
    """

    value: float

    def __post_init__(self) -> None:
        """Reject a non-positive threshold.

        Raises
        ------
        ValueError
            If ``value`` is not greater than 0.
        """
        if not self.value > 0:
            raise ValueError("threshold must be > 0")


@dataclass(frozen=True)
class AdaptiveDaily:
    """Target ``bars_per_day`` from the trailing mean daily measure (prior complete days only)."""

    bars_per_day: float
    lookback_days: int = 20
    initial: float | None = None  # threshold before any complete day exists
    min_days: int = 1  # complete days required before adapting (else ``initial``)
    floor: float = 0.0  # lower bound on the threshold

    def __post_init__(self) -> None:
        """Reject a non-positive target or a non-positive lookback.

        Raises
        ------
        ValueError
            If ``bars_per_day``, ``lookback_days``, or ``min_days`` is not positive.
        """
        if self.bars_per_day <= 0 or self.lookback_days < 1 or self.min_days < 1:
            raise ValueError("bad AdaptiveDaily parameters")


Threshold = Fixed | AdaptiveDaily


@dataclass
class EventBar:
    """One sealed or open information-driven bar for a single asset.

    Notes
    -----
    Prices are in the caller's price units. ``volume`` is the sum of trade sizes.
    ``dollar`` is the sum of price times size. ``n_ticks`` is the trade count.
    ``threshold`` is the threshold that was in force when the bar closed, or the
    current threshold if the bar is still open. Times are nanoseconds UTC.
    """

    start: int
    end: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar: float
    n_ticks: int
    threshold: float

    def as_tuple(self) -> tuple:
        """Return the fields in ``EVENT_FIELDS`` order.

        Returns
        -------
        tuple
            start, end, open, high, low, close, volume, dollar, n_ticks, threshold.
        """
        return tuple(getattr(self, k) for k in EVENT_FIELDS)


class EventBarBuilder:
    """Incremental per-asset event-bar builder. ``update`` returns the bars it sealed."""

    def __init__(
        self,
        kind: str = "volume",
        threshold: Threshold | float = 1000.0,
        overshoot: str = "close",
        day_key: Callable[[int], int] = utc_day,
    ):
        """Start an empty builder.

        Parameters
        ----------
        kind : str, optional
            ``tick``, ``volume``, or ``dollar``. Default is ``volume``.
        threshold : Fixed, AdaptiveDaily, or float, optional
            A float becomes ``Fixed(threshold)``. Default is 1000.
        overshoot : str, optional
            ``close`` (default) keeps the whole crossing trade in the bar it
            finishes. ``split`` cuts that trade so each bar holds exactly the
            threshold. Tick bars cannot split.
        day_key : callable, optional
            ``ts -> day id`` for adaptive thresholds. Default is :func:`utc_day`.

        Raises
        ------
        ValueError
            If ``kind`` or ``overshoot`` is unknown, or an adaptive rule has no ``initial`` threshold.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if overshoot not in OVERSHOOT:
            raise ValueError(f"overshoot must be one of {OVERSHOOT}")
        if kind == "tick" and overshoot == "split":
            overshoot = "close"  # a tick is indivisible; both policies coincide
        self.kind, self.overshoot, self.day_key = kind, overshoot, day_key
        self.rule: Threshold = Fixed(float(threshold)) if isinstance(threshold, (int, float)) else threshold
        if isinstance(self.rule, AdaptiveDaily) and self.rule.initial is None:
            raise ValueError("AdaptiveDaily needs an initial threshold for the first days")
        self._days: deque[float] = deque(maxlen=getattr(self.rule, "lookback_days", 1))
        self._day: int | None = None
        self._day_total = 0.0
        self.threshold = self._threshold()
        self._bar: EventBar | None = None
        self._cum = 0.0
        self.n_sealed = 0

    # ---------------------------------------------------------------- threshold
    def _threshold(self) -> float:
        r = self.rule
        if isinstance(r, Fixed):
            return r.value
        if len(self._days) < r.min_days:
            return float(r.initial)  # type: ignore[arg-type]
        avg = sum(self._days) / len(self._days)
        return max(avg / r.bars_per_day, r.floor) if avg > 0 else float(r.initial)  # type: ignore[arg-type]

    def _roll_day(self, ts: int) -> None:
        d = self.day_key(ts)
        if d == self._day:
            return
        if self._day is not None:
            # only days that had trades count (weekends/holidays are not zero-volume days)
            self._days.append(self._day_total)
        self._day, self._day_total = d, 0.0
        self.threshold = self._threshold()

    # ---------------------------------------------------------------- bars
    def _add(self, ts: int, price: float, size: float) -> None:
        b = self._bar
        if b is None:
            self._bar = EventBar(ts, ts, price, price, price, price, size, price * size, 1, self.threshold)
            return
        b.end = ts
        if price > b.high:
            b.high = price
        if price < b.low:
            b.low = price
        b.close = price
        b.volume += size
        b.dollar += price * size
        b.n_ticks += 1

    def _seal(self, out: list[EventBar]) -> None:
        assert self._bar is not None
        self._bar.threshold = self.threshold
        out.append(self._bar)
        self._bar = None
        self._cum = 0.0
        self.n_sealed += 1

    def _measure(self, price: float, size: float) -> float:
        if self.kind == "tick":
            return 1.0
        if self.kind == "volume":
            return size
        return price * size

    def update(self, ts: int, price: float, size: float) -> list[EventBar]:
        """Fold one trade into the open bar.

        Parameters
        ----------
        ts : int
            Trade time, nanoseconds UTC.
        price : float
            Trade price.
        size : float
            Trade size. The measure is 1, ``size``, or ``price * size`` depending on ``kind``.

        Returns
        -------
        list of EventBar
            The bars this trade sealed. Usually empty or one. ``split`` can
            seal several when one trade is larger than the threshold. The
            still-open bar is not included; read :attr:`open_bar`.
        """
        ts, price, size = int(ts), float(price), float(size)
        self._roll_day(ts)
        out: list[EventBar] = []
        m = self._measure(price, size)
        self._day_total += m
        thr = self.threshold
        if self.overshoot == "close" or m <= 0:
            self._add(ts, price, size)
            self._cum += m
            if self._cum >= thr:
                self._seal(out)
            return out
        # split: each bar takes exactly what it needs; the remainder carries on
        remaining = m
        eps = 1e-12 * thr
        while remaining > eps:
            need = thr - self._cum
            take = need if remaining >= need - eps else remaining
            frac = take / m
            self._add(ts, price, size * frac)
            self._cum += take
            remaining -= take
            if self._cum >= thr - eps:
                self._seal(out)
        return out

    def update_many(self, ts: Iterable[int], price: Iterable[float], size: Iterable[float]) -> list[EventBar]:
        """Fold a stream of trades, in order.

        Parameters
        ----------
        ts : iterable of int
            Trade times, nanoseconds UTC.
        price : iterable of float
            Trade prices.
        size : iterable of float
            Trade sizes. Must be the same length as ``ts`` and ``price``.

        Returns
        -------
        list of EventBar
            Every bar sealed by the stream, in order.

        Raises
        ------
        ValueError
            If the three iterables have different lengths.
        """
        out: list[EventBar] = []
        for t, p, q in zip(ts, price, size, strict=True):
            out.extend(self.update(t, p, q))
        return out

    @property
    def open_bar(self) -> EventBar | None:
        """The provisional bar (a copy), or None if no trades since the last seal."""
        if self._bar is None:
            return None
        b = self._bar
        return EventBar(
            b.start, b.end, b.open, b.high, b.low, b.close, b.volume, b.dollar, b.n_ticks, self.threshold
        )

    @property
    def progress(self) -> float:
        """Fraction of the threshold the open bar has filled."""
        return self._cum / self.threshold


@dataclass
class EventBars:
    """Struct-of-arrays event bars for one asset."""

    data: dict[str, np.ndarray]

    def __len__(self) -> int:
        """Return the number of sealed bars.

        Returns
        -------
        int
            Length of the ``end`` array.
        """
        return int(self.data["end"].size)

    def __getitem__(self, k: str) -> np.ndarray:
        """Return one field for every sealed bar.

        Parameters
        ----------
        k : str
            A field name: start, end, open, high, low, close, volume, dollar, n_ticks, threshold.

        Returns
        -------
        ndarray
            That field, one entry per bar.
        """
        return self.data[k]

    @classmethod
    def from_bars(cls, bars: list[EventBar]) -> EventBars:
        """Stack a list of bars into arrays.

        Parameters
        ----------
        bars : list of EventBar
            Sealed bars, in time order. May be empty.

        Returns
        -------
        EventBars
            One array per field.
        """
        dt = {"start": np.int64, "end": np.int64, "n_ticks": np.int64}
        return cls(
            {k: np.array([getattr(b, k) for b in bars], dtype=dt.get(k, np.float64)) for k in EVENT_FIELDS}
        )

    def panel(self, asset: str = "asset") -> Panel:
        """Return a one-asset panel whose rows are these event bars.

        Parameters
        ----------
        asset : str, optional
            The single asset name. Default is ``"asset"``.

        Returns
        -------
        Panel
            Rows are labeled by bar end. Columns are the OHLCV fields plus
            ``has_trade``, which is True on every sealed event bar.
        """
        cols = {
            k: self.data[k].astype(np.float64)[:, None]
            for k in ("open", "high", "low", "close", "volume", "dollar", "n_ticks")
        }
        cols["has_trade"] = np.ones((len(self), 1), dtype=bool)
        return Panel(self.data["end"], (asset,), cols, self.data["start"])


def build_event_bars(
    ts,
    price,
    size,
    kind: str = "volume",
    threshold: Threshold | float = 1000.0,
    overshoot: str = "close",
    day_key: Callable[[int], int] = utc_day,
) -> tuple[EventBars, EventBar | None]:
    """Batch: all sealed bars for one asset's trade stream, plus the open bar."""
    b = EventBarBuilder(kind, threshold, overshoot, day_key)
    bars = b.update_many(
        np.asarray(ts, np.int64).tolist(),
        np.asarray(price, np.float64).tolist(),
        np.asarray(size, np.float64).tolist(),
    )
    return EventBars.from_bars(bars), b.open_bar


# ====================================================================== clock alignment
def sample_asof(
    series_end: np.ndarray, values: np.ndarray, grid_ends: np.ndarray, missing: float = np.nan
) -> np.ndarray:
    """Return the last event value strictly before each grid-bar end.

    Parameters
    ----------
    series_end : ndarray of int64
        Event-bar ends, nanoseconds UTC, non-decreasing.
    values : ndarray
        One value per event bar, aligned with ``series_end``.
    grid_ends : ndarray of int64
        Clock-bar ends, nanoseconds UTC.
    missing : float, optional
        Value used when no event bar ends before the grid bar. Default is NaN.

    Returns
    -------
    ndarray of float64
        Shape ``(len(grid_ends),)``. The comparison is strict (``end < grid_end``)
        so a bar sealed by a trade at ``ts`` is visible in the clock bar that
        contains ``ts``, matching the ``[start, end)`` clock convention.
    """
    grid_ends = np.asarray(grid_ends, np.int64)
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.full(grid_ends.size, missing, dtype=np.float64)
    idx = np.searchsorted(np.asarray(series_end, np.int64), grid_ends, side="left") - 1
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], missing).astype(np.float64)


def sample_panel(
    per_asset: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    grid_ends: np.ndarray,
    assets: tuple[str, ...] | None = None,
) -> Panel:
    """Batch as-of sampling of per-asset event series onto a clock grid -> :class:`Panel`.

    ``per_asset[asset] = (end_times, {column: values})``.
    """
    names = tuple(assets) if assets is not None else tuple(per_asset)
    grid_ends = np.asarray(grid_ends, np.int64)
    cols: dict[str, np.ndarray] = {}
    for j, a in enumerate(names):
        ends, vals = per_asset[a]
        for k, v in vals.items():
            arr = cols.setdefault(k, np.full((grid_ends.size, len(names)), np.nan))
            arr[:, j] = sample_asof(ends, v, grid_ends)
    return Panel(grid_ends, names, cols)


class AsofSampler:
    """Incremental as-of sampling: feed sealed event rows, read a clock-grid row at any time."""

    def __init__(self, assets: tuple[str, ...], columns: tuple[str, ...]):
        """Prepare an empty sampler.

        Parameters
        ----------
        assets : tuple of str
            Column order of every row this sampler returns.
        columns : tuple of str
            Value names each pushed row may contain.
        """
        self.assets = tuple(assets)
        self.index = {a: i for i, a in enumerate(self.assets)}
        self.columns = tuple(columns)
        self._rows: list[deque[tuple[int, dict[str, float]]]] = [deque() for _ in self.assets]

    def push(self, asset: str, end: int, values: dict[str, float]) -> None:
        """Record one sealed event bar.

        Parameters
        ----------
        asset : str
            Must be one of the assets passed to the constructor.
        end : int
            Bar end, nanoseconds UTC. Must be at least the previous end for this asset.
        values : dict of str to float
            Column values. Missing names become NaN when sampled.

        Raises
        ------
        ValueError
            If ``end`` is earlier than the previous bar for this asset.
        KeyError
            If ``asset`` was not declared.
        """
        q = self._rows[self.index[asset]]
        if q and end < q[-1][0]:
            raise ValueError("event rows must arrive in end-time order")
        q.append((int(end), dict(values)))

    def row(self, grid_end: int) -> dict[str, np.ndarray]:
        """Return values as of the last sealed event bar before ``grid_end``.

        Parameters
        ----------
        grid_end : int
            Clock-bar end, nanoseconds UTC. Call with non-decreasing values:
            rows older than the one used are dropped.

        Returns
        -------
        dict of str to ndarray
            One ``(N,)`` array per column. The sample is the last event row
            with ``end < grid_end``. Assets with nothing that old are NaN.
        """
        out = {k: np.full(len(self.assets), np.nan) for k in self.columns}
        for j, q in enumerate(self._rows):
            while len(q) > 1 and q[1][0] < grid_end:
                q.popleft()
            if q and q[0][0] < grid_end:
                for k in self.columns:
                    out[k][j] = q[0][1].get(k, np.nan)
        return out


class EventPipeline:
    """A per-asset pipeline driven by event bars: each sealed event bar seals one row.

    The open event bar is the pipeline's open row, so ``update`` returns provisional values
    (``is_open=True``) between bar closes, exactly like the clock-time engine.
    """

    def __init__(
        self,
        pipeline: Pipeline | Graph,
        builder: EventBarBuilder,
        history: EventBars | None = None,
        asset: str = "asset",
    ):
        """Warm a one-asset live engine from sealed event bars.

        Parameters
        ----------
        pipeline : Pipeline or Graph
            Factors to run. Inputs are the usual bar fields.
        builder : EventBarBuilder
            The bar state machine. If it already holds an open bar, that bar
            is the pipeline's open row immediately.
        history : EventBars or None, optional
            Sealed bars to warm from. ``None`` starts empty, so windows stay missing until enough bars seal.
        asset : str, optional
            Name of the single asset. Default is ``"asset"``.
        """
        self.builder = builder
        hist = history.panel(asset) if history is not None else EventBars.from_bars([]).panel(asset)
        self.engine = LiveEngine(pipeline, hist)
        self.asset = asset
        if builder.open_bar is not None:  # a builder resumed mid-bar: its open bar is the open row
            self._write_open(builder.open_bar)

    def revise(self) -> Snapshot:
        """Provisional values for the current open event bar without feeding a trade."""
        return self.engine.revise()

    def _write_open(self, b: EventBar | None) -> None:
        f = self.engine.bar
        if b is None:
            f.start(f.f["close"].copy())
            self.engine._dirty[:] = True
            return
        self.engine.set_bar(0, b.open, b.high, b.low, b.close, b.volume, b.dollar, float(b.n_ticks))

    def update(self, ts: int, price: float, size: float) -> list[Snapshot]:
        """Feed one trade. Returns a sealed snapshot per closed bar, then the open revision."""
        out: list[Snapshot] = []
        for bar in self.builder.update(ts, price, size):
            self._write_open(bar)
            out.append(self.engine.seal(bar.end))
        ob = self.builder.open_bar
        if ob is not None:
            self._write_open(ob)
            out.append(self.engine.revise())
        return out
