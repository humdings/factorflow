"""Panel: dense ``T × N`` arrays on a bar grid.

Rows are bars, labeled by end time. Columns are named arrays of shape ``(T, N)``
with one column per asset. Missing factor values are NaN. Classifier codes use
the column's own dtype (``-1`` is the engine's missing classifier code when the
column is read as a :class:`~factorflow.terms.Classifier`).

Examples
--------
>>> import numpy as np
>>> from factorflow import Panel
>>> p = Panel(np.arange(3), ("AAA", "BBB"), {"close": np.ones((3, 2))})
>>> p.shape
(3, 2)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class Panel:
    """One history: bar-end times, asset names, and dense columns.

    Parameters
    ----------
    times : array of int64
        Bar end, nanoseconds since the Unix epoch, UTC. Must be non-decreasing.
        Event bars may share a timestamp. Length ``T``.
    assets : sequence of str
        Asset names, length ``N``. Order is the column order of every array.
    columns : dict of str to ndarray, optional
        Each array has shape ``(T, N)``. Omitted columns are simply absent;
        a pipeline that reads one will fail at runtime with ``KeyError``.
    starts : array of int64 or None, optional
        Bar start, same clock and length as ``times``. ``None`` if the caller
        does not have starts.

    Raises
    ------
    ValueError
        If ``times`` decreases, or a column's shape is not ``(T, N)``.
    """

    times: np.ndarray  # int64 ns since epoch (UTC), bar end; non-decreasing (event bars may share a stamp)
    assets: tuple[str, ...]
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    starts: np.ndarray | None = None  # optional bar start times

    def __post_init__(self) -> None:
        """Check shapes and that bar times do not go backwards.

        Raises
        ------
        ValueError
            If ``times`` is not non-decreasing, or a column shape is not ``(T, N)``.
        """
        self.times = np.asarray(self.times, dtype=np.int64)
        self.assets = tuple(self.assets)
        t, n = len(self.times), len(self.assets)
        if t > 1 and not np.all(np.diff(self.times) >= 0):
            raise ValueError("panel times must be non-decreasing")
        for k, v in self.columns.items():
            if v.shape != (t, n):
                raise ValueError(f"column {k!r} has shape {v.shape}, expected {(t, n)}")

    @property
    def shape(self) -> tuple[int, int]:
        """Number of bars and number of assets.

        Returns
        -------
        tuple of int
            ``(T, N)``.
        """
        return len(self.times), len(self.assets)

    def head(self, rows: int) -> Panel:
        """Return the first ``rows`` bars.

        Parameters
        ----------
        rows : int
            How many leading bars to keep. Not a time span.

        Returns
        -------
        Panel
            A new panel. Arrays are views of the originals.
        """
        return Panel(
            self.times[:rows],
            self.assets,
            {k: v[:rows] for k, v in self.columns.items()},
            None if self.starts is None else self.starts[:rows],
        )

    def with_row(self, row: Mapping[str, np.ndarray], time: int) -> Panel:
        """Return a copy with one bar appended.

        Used to treat an open bar as if it were already sealed, which is how a
        live revision is checked against :class:`~factorflow.batch.BatchEngine`.

        Parameters
        ----------
        row : mapping of str to ndarray
            One value per asset for every column this panel already has.
            Dtypes follow the existing columns.
        time : int
            End time of the new bar, nanoseconds UTC.

        Returns
        -------
        Panel
            ``T + 1`` bars. ``starts`` is dropped.
        """
        cols = {
            k: np.vstack([v, np.asarray(row[k], dtype=v.dtype)[None, :]]) for k, v in self.columns.items()
        }
        return Panel(np.r_[self.times, np.int64(time)], self.assets, cols)

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        assets: Sequence[str] | None = None,
        time_col: str = "ts",
        asset_col: str = "asset",
    ) -> Panel:
        """Build a panel from a long polars frame.

        Parameters
        ----------
        df : polars.DataFrame
            One row per ``(ts, asset)``. Other columns become panel columns.
            ``ts`` may be int64 nanoseconds or a UTC datetime. Missing pairs
            become NaN. Non-float columns are cast to float64.
        assets : sequence of str or None, optional
            Column order. ``None`` sorts the names that appear in ``df``.
        time_col : str, optional
            Timestamp column. Default is ``"ts"``.
        asset_col : str, optional
            Asset-name column. Default is ``"asset"``.

        Returns
        -------
        Panel
            Dense ``(T, N)`` columns. Assets in ``df`` that are not in
            ``assets`` are dropped.

        Examples
        --------
        >>> import polars as pl
        >>> from factorflow import Panel
        >>> df = pl.DataFrame({"ts": [0, 0], "asset": ["A", "B"], "close": [1.0, 2.0]})
        >>> Panel.from_frame(df).assets
        ('A', 'B')
        """
        ts = df[time_col]
        if ts.dtype != pl.Int64:
            ts = ts.cast(pl.Datetime("ns", "UTC")).cast(pl.Int64)
        df = df.with_columns(ts.alias("__t"))
        times = np.unique(df["__t"].to_numpy())
        names = tuple(assets) if assets is not None else tuple(sorted(df[asset_col].unique().to_list()))
        ti = np.searchsorted(times, df["__t"].to_numpy())
        amap = {a: i for i, a in enumerate(names)}
        ai = np.array([amap.get(a, -1) for a in df[asset_col].to_list()])
        keep = ai >= 0
        cols = {}
        for c in df.columns:
            if c in (time_col, asset_col, "__t"):
                continue
            arr = np.full((len(times), len(names)), np.nan)
            arr[ti[keep], ai[keep]] = df[c].to_numpy()[keep].astype(np.float64)
            cols[c] = arr
        return cls(times, names, cols)
