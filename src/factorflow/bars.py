"""Clock-time bar construction (trades -> OHLCV bars on a :class:`BarGrid`).

One accumulator, :class:`BarState`, is shared by the batch builder and the live engine,
so a bar built live from a trade stream is identical to the batch-built bar.

Bar rules (docs/BARS.md):

* a trade at ``ts`` belongs to the first bar with ``end > ts`` (bars are ``[start, end)``,
  labeled by ``end``);
* a bar with no trades carries the previous close forward: ``open = high = low = close =
  previous close`` and ``volume = dollar = n_ticks = 0`` (NaN before an asset's first trade);
* ``has_trade`` is ``n_ticks > 0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from factorflow.panel import Panel

FIELDS = ("open", "high", "low", "close", "volume", "dollar", "n_ticks")


@dataclass
class Trades:
    """A trade stream: parallel arrays sorted by ``ts`` (int64 ns UTC).

    Parameters
    ----------
    asset : array of int
        Index into the asset list passed to :func:`build_time_bars`.
    ts : array of int64
        Trade time, nanoseconds UTC, non-decreasing.
    price : array of float
        Trade price, in the same units the caller uses everywhere else.
    size : array of float
        Trade size. The sign is not special: volume bars add ``size`` as given.

    Raises
    ------
    ValueError
        If ``ts`` goes backwards.
    """

    asset: np.ndarray  # int asset index
    ts: np.ndarray
    price: np.ndarray
    size: np.ndarray

    def __post_init__(self) -> None:
        """Cast the arrays and require non-decreasing timestamps.

        Raises
        ------
        ValueError
            If a timestamp is earlier than the one before it.
        """
        self.asset = np.asarray(self.asset, dtype=np.int64)
        self.ts = np.asarray(self.ts, dtype=np.int64)
        self.price = np.asarray(self.price, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)
        if self.ts.size > 1 and not np.all(np.diff(self.ts) >= 0):
            raise ValueError("trades must be sorted by ts")

    def __len__(self) -> int:
        """Return the number of trades."""
        return int(self.ts.size)


class BarState:
    """The open (provisional) bar for ``N`` assets.

    Batch (:func:`build_time_bars`) and live (:class:`~factorflow.live.LiveEngine`)
    both fold trades through this object, so a bar built either way is the same
    object updated by the same rules.

    Parameters
    ----------
    n : int
        Number of assets.
    """

    def __init__(self, n: int):
        """Allocate one open bar and start it from a missing previous close.

        Parameters
        ----------
        n : int
            Number of assets.
        """
        self.n = n
        self.f = {k: np.full(n, np.nan) for k in FIELDS}
        self.start(np.full(n, np.nan))

    def start(self, prev_close: np.ndarray) -> None:
        """Open a new bar from the previous close.

        Parameters
        ----------
        prev_close : ndarray of shape (N,)
            Close to carry forward. OHLC start at this value. Volume, dollar
            value, and trade count start at 0. A NaN previous close stays NaN
            until the first trade, which is an asset that has not traded yet.
        """
        pc = np.asarray(prev_close, dtype=np.float64)
        for k in ("open", "high", "low", "close"):
            self.f[k][:] = pc
        for k in ("volume", "dollar", "n_ticks"):
            self.f[k][:] = 0.0

    def tick(self, j: int, price: float, size: float) -> None:
        """Apply one trade to asset ``j``.

        Parameters
        ----------
        j : int
            Asset column.
        price : float
            Trade price. The first trade in the bar sets open, high, and low.
            Later trades expand high and low and replace the close.
        size : float
            Added to volume. Dollar volume increases by ``price * size``.
            The trade count increases by 1.
        """
        f = self.f
        if f["n_ticks"][j] == 0:
            f["open"][j] = f["high"][j] = f["low"][j] = price
        else:
            if price > f["high"][j]:
                f["high"][j] = price
            if price < f["low"][j]:
                f["low"][j] = price
        f["close"][j] = price
        f["volume"][j] += size
        f["dollar"][j] += price * size
        f["n_ticks"][j] += 1

    def set_bar(
        self,
        j: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        dollar: float | None = None,
        n_ticks: float | None = None,
    ) -> None:
        """Replace asset ``j``'s open bar with values the caller already aggregated.

        Parameters
        ----------
        j : int
            Asset column.
        open, high, low, close, volume : float
            The in-progress bar. Prices are in the caller's price units.
            ``volume`` is in the caller's size units.
        dollar : float or None, optional
            Price times size accumulated so far. ``None`` stores ``close * volume``.
        n_ticks : float or None, optional
            Trade count. ``None`` stores 1 when ``volume > 0``, else 0.

        Notes
        -----
        This is the path for a feed that publishes the current bar instead of
        individual trades. It does not check that high and low bracket the close.
        """
        f = self.f
        f["open"][j], f["high"][j], f["low"][j], f["close"][j] = open, high, low, close
        f["volume"][j] = volume
        f["dollar"][j] = close * volume if dollar is None else dollar
        f["n_ticks"][j] = (1.0 if volume > 0 else 0.0) if n_ticks is None else n_ticks

    def row(self, name: str) -> np.ndarray:
        """Return one field of the open bar.

        Parameters
        ----------
        name : str
            ``"open"``, ``"high"``, ``"low"``, ``"close"``, ``"volume"``,
            ``"dollar"``, ``"n_ticks"``, or ``"has_trade"``. ``has_trade`` is
            ``n_ticks > 0`` and is not stored.

        Returns
        -------
        ndarray of shape (N,)
            A view of the field, except ``has_trade``, which is a new bool array.
        """
        if name == "has_trade":
            return self.f["n_ticks"] > 0
        return self.f[name]


def build_time_bars(
    trades: Trades,
    assets: Sequence[str],
    starts: np.ndarray,
    ends: np.ndarray,
    prev_close: np.ndarray | None = None,
) -> Panel:
    """Aggregate trades into clock bars.

    Parameters
    ----------
    trades : Trades
        Sorted by timestamp. ``trades.asset`` indexes ``assets``. A trade
        belongs to the first bar with ``end > ts``.
    assets : sequence of str
        Asset names, length ``N``.
    starts, ends : ndarray of int64
        Bar bounds, nanoseconds UTC, from :meth:`BarGrid.bars`. Same length.
    prev_close : ndarray of shape (N,) or None, optional
        Close before the first bar. ``None`` starts from NaN, so an asset
        with no trade in the first bars stays missing.

    Returns
    -------
    Panel
        Columns ``open``, ``high``, ``low``, ``close``, ``volume``, ``dollar``,
        ``n_ticks``, and bool ``has_trade``. An empty bar repeats the previous
        close and stores zero volume. ``times`` is ``ends``.

    Notes
    -----
    The loop is the same :class:`BarState` the live engine ticks. Trades must
    be in time order. Out-of-order trades are not repaired here.
    """
    ends = np.asarray(ends, dtype=np.int64)
    t, n = len(ends), len(assets)
    out = {k: np.empty((t, n)) for k in FIELDS}
    st = BarState(n)
    st.start(np.full(n, np.nan) if prev_close is None else prev_close)
    bar_of = np.searchsorted(ends, trades.ts, side="right")
    cuts = np.searchsorted(bar_of, np.arange(t + 1), side="left")
    for r in range(t):
        for i in range(cuts[r], cuts[r + 1]):
            st.tick(int(trades.asset[i]), float(trades.price[i]), float(trades.size[i]))
        for k in FIELDS:
            out[k][r] = st.f[k]
        st.start(st.f["close"].copy())
    cols: dict[str, np.ndarray] = dict(out)
    cols["has_trade"] = out["n_ticks"] > 0
    return Panel(ends, tuple(assets), cols, np.asarray(starts, dtype=np.int64))
