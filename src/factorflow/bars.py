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
    """A trade stream: parallel arrays sorted by ``ts`` (int64 ns UTC)."""

    asset: np.ndarray  # int asset index
    ts: np.ndarray
    price: np.ndarray
    size: np.ndarray

    def __post_init__(self) -> None:
        self.asset = np.asarray(self.asset, dtype=np.int64)
        self.ts = np.asarray(self.ts, dtype=np.int64)
        self.price = np.asarray(self.price, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)
        if self.ts.size > 1 and not np.all(np.diff(self.ts) >= 0):
            raise ValueError("trades must be sorted by ts")

    def __len__(self) -> int:
        return int(self.ts.size)


class BarState:
    """The open (provisional) bar for N assets."""

    def __init__(self, n: int):
        self.n = n
        self.f = {k: np.full(n, np.nan) for k in FIELDS}
        self.start(np.full(n, np.nan))

    def start(self, prev_close: np.ndarray) -> None:
        pc = np.asarray(prev_close, dtype=np.float64)
        for k in ("open", "high", "low", "close"):
            self.f[k][:] = pc
        for k in ("volume", "dollar", "n_ticks"):
            self.f[k][:] = 0.0

    def tick(self, j: int, price: float, size: float) -> None:
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
        """Overwrite an asset's open bar (e.g. from a broker's in-progress bar update)."""
        f = self.f
        f["open"][j], f["high"][j], f["low"][j], f["close"][j] = open, high, low, close
        f["volume"][j] = volume
        f["dollar"][j] = close * volume if dollar is None else dollar
        f["n_ticks"][j] = (1.0 if volume > 0 else 0.0) if n_ticks is None else n_ticks

    def row(self, name: str) -> np.ndarray:
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
    """Batch: aggregate a trade stream onto grid bars (``starts``/``ends`` int64 ns)."""
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
